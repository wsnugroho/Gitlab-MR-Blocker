import logging
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

JIRA_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
SUPPORTED_MR_ACTIONS = {"open", "reopen", "update"}


class ConfigurationError(RuntimeError):
    """Raised when required environment variables are missing."""


class JiraApiError(RuntimeError):
    """Raised when Jira API interaction fails."""


class GitLabApiError(RuntimeError):
    """Raised when GitLab API interaction fails."""


def summarize_response_text(text: str, limit: int = 500) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


@dataclass(frozen=True)
class Settings:
    gitlab_token: str | None
    gitlab_url: str
    gitlab_webhook_secret: str | None
    jira_url: str | None
    jira_email: str | None
    jira_api_token: str | None
    jira_block_link_type_name: str
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        timeout_value = os.getenv("REQUEST_TIMEOUT_SECONDS", "10")
        return cls(
            gitlab_token=os.getenv("GITLAB_TOKEN"),
            gitlab_url=os.getenv("GITLAB_URL", "https://gitlab.cs.ui.ac.id/api/v4").rstrip("/"),
            gitlab_webhook_secret=os.getenv("GITLAB_WEBHOOK_SECRET"),
            jira_url=os.getenv("JIRA_URL", "").rstrip("/") or None,
            jira_email=os.getenv("JIRA_EMAIL"),
            jira_api_token=os.getenv("JIRA_API_TOKEN"),
            jira_block_link_type_name=os.getenv("JIRA_BLOCK_LINK_TYPE_NAME", "Blocks"),
            request_timeout_seconds=float(timeout_value),
        )

    def validate(self) -> None:
        missing = []
        if not self.gitlab_token:
            missing.append("GITLAB_TOKEN")
        if not self.jira_url:
            missing.append("JIRA_URL")
        if not self.jira_email:
            missing.append("JIRA_EMAIL")
        if not self.jira_api_token:
            missing.append("JIRA_API_TOKEN")

        if missing:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


@dataclass(frozen=True)
class MergeRequestEvent:
    project_id: int
    mr_iid: int
    title: str
    action: str

    @property
    def jira_key(self) -> str | None:
        return extract_jira_key(self.title)


@dataclass(frozen=True)
class IssueRelations:
    blockers: tuple[str, ...]
    blocked: tuple[str, ...]


@dataclass(frozen=True)
class MergeRequestRef:
    project_id: int
    iid: int
    title: str


@dataclass(frozen=True)
class MergeRequestResolution:
    jira_key: str
    status: str
    candidate: MergeRequestRef | None = None
    candidates: tuple[MergeRequestRef, ...] = ()


@dataclass(frozen=True)
class DependencyRecord:
    direction: str
    blocked_mr_iid: int
    blocking_mr_iid: int
    issue_key: str


@dataclass
class SyncReport:
    status: str
    action: str
    project_id: int
    mr_iid: int
    jira_key: str
    blockers_in_jira: list[str] = field(default_factory=list)
    blocked_issues_in_jira: list[str] = field(default_factory=list)
    created_dependencies: list[DependencyRecord] = field(default_factory=list)
    existing_dependencies: list[DependencyRecord] = field(default_factory=list)
    unresolved_blockers: list[str] = field(default_factory=list)
    unresolved_blocked_targets: list[str] = field(default_factory=list)
    ambiguous_blockers: dict[str, list[int]] = field(default_factory=dict)
    ambiguous_blocked_targets: dict[str, list[int]] = field(default_factory=dict)

    def add_created_dependency(
        self, direction: str, blocked_mr_iid: int, blocking_mr_iid: int, issue_key: str
    ) -> None:
        self.created_dependencies.append(
            DependencyRecord(
                direction=direction,
                blocked_mr_iid=blocked_mr_iid,
                blocking_mr_iid=blocking_mr_iid,
                issue_key=issue_key,
            )
        )

    def add_existing_dependency(
        self, direction: str, blocked_mr_iid: int, blocking_mr_iid: int, issue_key: str
    ) -> None:
        self.existing_dependencies.append(
            DependencyRecord(
                direction=direction,
                blocked_mr_iid=blocked_mr_iid,
                blocking_mr_iid=blocking_mr_iid,
                issue_key=issue_key,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "project_id": self.project_id,
            "mr_iid": self.mr_iid,
            "jira_key": self.jira_key,
            "blockers_in_jira": self.blockers_in_jira,
            "blocked_issues_in_jira": self.blocked_issues_in_jira,
            "created_dependencies": [asdict(item) for item in self.created_dependencies],
            "existing_dependencies": [asdict(item) for item in self.existing_dependencies],
            "unresolved_blockers": self.unresolved_blockers,
            "unresolved_blocked_targets": self.unresolved_blocked_targets,
            "ambiguous_blockers": self.ambiguous_blockers,
            "ambiguous_blocked_targets": self.ambiguous_blocked_targets,
        }


def extract_jira_key(title: str) -> str | None:
    match = JIRA_KEY_PATTERN.search(title or "")
    if not match:
        return None
    return match.group(1)


def parse_merge_request_event(payload: dict[str, Any]) -> MergeRequestEvent | None:
    if payload.get("object_kind") != "merge_request":
        return None

    attributes = payload.get("object_attributes") or {}
    action = attributes.get("action")
    if action not in SUPPORTED_MR_ACTIONS:
        return None

    project = payload.get("project") or {}
    project_id = project.get("id")
    mr_iid = attributes.get("iid")
    title = attributes.get("title") or ""

    if project_id is None or mr_iid is None:
        raise ValueError("Invalid merge request webhook payload")

    return MergeRequestEvent(
        project_id=int(project_id),
        mr_iid=int(mr_iid),
        title=title,
        action=action,
    )


class JiraClient:
    def __init__(self, http_client: httpx.AsyncClient, settings: Settings):
        self._http_client = http_client
        self._settings = settings

    async def get_issue_relations(self, issue_key: str) -> IssueRelations:
        response = await self._http_client.get(
            f"{self._settings.jira_url}/rest/api/3/issue/{issue_key}",
            auth=(self._settings.jira_email, self._settings.jira_api_token),
            headers={"Accept": "application/json"},
            params={"fields": "issuelinks"},
        )

        logger.info(
            "Jira issue fetch issue=%s status=%s",
            issue_key,
            response.status_code,
        )
        if response.status_code == status.HTTP_404_NOT_FOUND:
            raise JiraApiError(f"Jira issue {issue_key} not found")
        if response.is_error:
            logger.error(
                "Jira issue fetch failed issue=%s status=%s body=%s",
                issue_key,
                response.status_code,
                summarize_response_text(response.text),
            )
            raise JiraApiError(
                f"Failed to fetch Jira issue {issue_key}: "
                f"{response.status_code} {response.text}"
            )

        issue_links = response.json().get("fields", {}).get("issuelinks", [])
        blockers: set[str] = set()
        blocked: set[str] = set()

        for link in issue_links:
            link_type = link.get("type") or {}
            if not self._is_block_link_type(link_type):
                continue

            inward_issue = link.get("inwardIssue") or {}
            outward_issue = link.get("outwardIssue") or {}

            inward_key = inward_issue.get("key")
            outward_key = outward_issue.get("key")

            if inward_key:
                blockers.add(inward_key)
            if outward_key:
                blocked.add(outward_key)

        return IssueRelations(
            blockers=tuple(sorted(blockers)),
            blocked=tuple(sorted(blocked)),
        )

    def _is_block_link_type(self, link_type: dict[str, Any]) -> bool:
        configured_name = self._settings.jira_block_link_type_name.strip().lower()
        link_name = str(link_type.get("name", "")).strip().lower()
        inward = str(link_type.get("inward", "")).strip().lower()
        outward = str(link_type.get("outward", "")).strip().lower()

        return link_name == configured_name or (
            inward == "is blocked by" and outward == "blocks"
        )


class GitLabClient:
    def __init__(self, http_client: httpx.AsyncClient, settings: Settings):
        self._http_client = http_client
        self._settings = settings

    async def resolve_open_merge_request(
        self, project_id: int, jira_key: str, exclude_iid: int | None = None
    ) -> MergeRequestResolution:
        search_url = f"{self._settings.gitlab_url}/projects/{project_id}/merge_requests"
        params = {
            "state": "opened",
            "search": jira_key,
            "in": "title",
            "per_page": 100,
        }
        response = await self._http_client.get(
            search_url,
            headers=self._headers,
            params=params,
        )

        logger.info(
            "GitLab MR search project_id=%s jira_key=%s status=%s",
            project_id,
            jira_key,
            response.status_code,
        )
        if response.is_error:
            logger.error(
                "GitLab MR search failed project_id=%s jira_key=%s status=%s body=%s",
                project_id,
                jira_key,
                response.status_code,
                summarize_response_text(response.text),
            )
            raise GitLabApiError(
                "Failed to search merge requests for "
                f"{jira_key}: {response.status_code} {response.text}"
            )

        exact_matches = []
        for item in response.json():
            candidate = MergeRequestRef(
                project_id=int(item["project_id"]),
                iid=int(item["iid"]),
                title=item.get("title", ""),
            )
            if exclude_iid is not None and candidate.iid == exclude_iid:
                continue
            if extract_jira_key(candidate.title) == jira_key:
                exact_matches.append(candidate)

        if not exact_matches:
            return MergeRequestResolution(jira_key=jira_key, status="unresolved")
        if len(exact_matches) > 1:
            return MergeRequestResolution(
                jira_key=jira_key,
                status="ambiguous",
                candidates=tuple(exact_matches),
            )
        return MergeRequestResolution(
            jira_key=jira_key,
            status="resolved",
            candidate=exact_matches[0],
        )

    async def get_dependencies(self, project_id: int, mr_iid: int) -> dict[int, int]:
        dependency_url = (
            f"{self._settings.gitlab_url}/projects/{project_id}/merge_requests/{mr_iid}/blocks"
        )
        response = await self._http_client.get(
            dependency_url,
            headers=self._headers,
        )

        logger.info(
            "GitLab dependency fetch project_id=%s blocked_mr_iid=%s status=%s",
            project_id,
            mr_iid,
            response.status_code,
        )
        if response.is_error:
            logger.error(
                "GitLab dependency fetch failed project_id=%s blocked_mr_iid=%s status=%s body=%s",
                project_id,
                mr_iid,
                response.status_code,
                summarize_response_text(response.text),
            )
            raise GitLabApiError(
                "Failed to fetch merge request dependencies for "
                f"!{mr_iid}: {response.status_code} {response.text}"
            )

        dependencies = {}
        for item in response.json():
            blocking_merge_request = item.get("blocking_merge_request") or {}
            blocking_iid = blocking_merge_request.get("iid")
            block_id = item.get("id")
            if blocking_iid is None or block_id is None:
                continue
            dependencies[int(blocking_iid)] = int(block_id)

        return dependencies

    async def create_dependency(
        self, project_id: int, blocked_mr_iid: int, blocking_merge_request: MergeRequestRef
    ) -> bool:
        params: dict[str, Any] = {
            "blocking_merge_request_iid": blocking_merge_request.iid,
        }
        if blocking_merge_request.project_id != project_id:
            params["blocking_project_id"] = blocking_merge_request.project_id

        dependency_url = (
            f"{self._settings.gitlab_url}/projects/{project_id}/merge_requests/{blocked_mr_iid}/blocks"
        )
        response = await self._http_client.post(
            dependency_url,
            headers=self._headers,
            params=params,
        )

        logger.info(
            "GitLab dependency create project_id=%s blocked_mr_iid=%s blocking_project_id=%s blocking_mr_iid=%s status=%s",
            project_id,
            blocked_mr_iid,
            blocking_merge_request.project_id,
            blocking_merge_request.iid,
            response.status_code,
        )
        if response.status_code == status.HTTP_201_CREATED:
            return True
        if response.status_code == status.HTTP_409_CONFLICT:
            return False

        logger.error(
            "GitLab dependency create failed project_id=%s blocked_mr_iid=%s blocking_project_id=%s blocking_mr_iid=%s status=%s body=%s",
            project_id,
            blocked_mr_iid,
            blocking_merge_request.project_id,
            blocking_merge_request.iid,
            response.status_code,
            summarize_response_text(response.text),
        )
        raise GitLabApiError(
            "Failed to create merge request dependency "
            f"!{blocked_mr_iid} <- !{blocking_merge_request.iid}: "
            f"{response.status_code} {response.text}"
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {"PRIVATE-TOKEN": self._settings.gitlab_token}


class DependencySyncService:
    def __init__(self, jira_client: JiraClient, gitlab_client: GitLabClient):
        self._jira_client = jira_client
        self._gitlab_client = gitlab_client
        self._resolution_cache: dict[tuple[int, str, int | None], MergeRequestResolution] = {}
        self._dependency_cache: dict[tuple[int, int], dict[int, int]] = {}

    async def sync(self, event: MergeRequestEvent) -> SyncReport:
        jira_key = event.jira_key
        if not jira_key:
            return SyncReport(
                status="skipped",
                action=event.action,
                project_id=event.project_id,
                mr_iid=event.mr_iid,
                jira_key="",
            )

        relations = await self._jira_client.get_issue_relations(jira_key)
        report = SyncReport(
            status="success",
            action=event.action,
            project_id=event.project_id,
            mr_iid=event.mr_iid,
            jira_key=jira_key,
            blockers_in_jira=list(relations.blockers),
            blocked_issues_in_jira=list(relations.blocked),
        )

        await self._sync_inbound_dependencies(event, relations, report)
        await self._sync_outbound_dependencies(event, relations, report)
        return report

    async def _sync_inbound_dependencies(
        self, event: MergeRequestEvent, relations: IssueRelations, report: SyncReport
    ) -> None:
        for blocker_key in relations.blockers:
            resolution = await self._resolve_merge_request(
                event.project_id, blocker_key, exclude_iid=event.mr_iid
            )
            if resolution.status == "unresolved":
                report.unresolved_blockers.append(blocker_key)
                continue
            if resolution.status == "ambiguous":
                report.ambiguous_blockers[blocker_key] = [
                    candidate.iid for candidate in resolution.candidates
                ]
                continue

            await self._ensure_dependency(
                project_id=event.project_id,
                blocked_mr_iid=event.mr_iid,
                blocking_merge_request=resolution.candidate,
                issue_key=blocker_key,
                direction="inbound",
                report=report,
            )

    async def _sync_outbound_dependencies(
        self, event: MergeRequestEvent, relations: IssueRelations, report: SyncReport
    ) -> None:
        current_merge_request = MergeRequestRef(
            project_id=event.project_id,
            iid=event.mr_iid,
            title=event.title,
        )

        for blocked_issue_key in relations.blocked:
            resolution = await self._resolve_merge_request(
                event.project_id, blocked_issue_key, exclude_iid=event.mr_iid
            )
            if resolution.status == "unresolved":
                report.unresolved_blocked_targets.append(blocked_issue_key)
                continue
            if resolution.status == "ambiguous":
                report.ambiguous_blocked_targets[blocked_issue_key] = [
                    candidate.iid for candidate in resolution.candidates
                ]
                continue

            await self._ensure_dependency(
                project_id=event.project_id,
                blocked_mr_iid=resolution.candidate.iid,
                blocking_merge_request=current_merge_request,
                issue_key=event.jira_key,
                direction="outbound",
                report=report,
            )

    async def _resolve_merge_request(
        self, project_id: int, jira_key: str, exclude_iid: int | None = None
    ) -> MergeRequestResolution:
        cache_key = (project_id, jira_key, exclude_iid)
        if cache_key not in self._resolution_cache:
            self._resolution_cache[cache_key] = await self._gitlab_client.resolve_open_merge_request(
                project_id=project_id,
                jira_key=jira_key,
                exclude_iid=exclude_iid,
            )
        return self._resolution_cache[cache_key]

    async def _ensure_dependency(
        self,
        project_id: int,
        blocked_mr_iid: int,
        blocking_merge_request: MergeRequestRef | None,
        issue_key: str,
        direction: str,
        report: SyncReport,
    ) -> None:
        if blocking_merge_request is None:
            return
        if blocked_mr_iid == blocking_merge_request.iid:
            return

        dependencies = await self._get_dependency_map(project_id, blocked_mr_iid)
        if blocking_merge_request.iid in dependencies:
            report.add_existing_dependency(
                direction=direction,
                blocked_mr_iid=blocked_mr_iid,
                blocking_mr_iid=blocking_merge_request.iid,
                issue_key=issue_key,
            )
            return

        created = await self._gitlab_client.create_dependency(
            project_id=project_id,
            blocked_mr_iid=blocked_mr_iid,
            blocking_merge_request=blocking_merge_request,
        )
        dependencies[blocking_merge_request.iid] = -1

        if created:
            logger.info(
                "Created dependency direction=%s blocked=!%s blocking=!%s issue=%s",
                direction,
                blocked_mr_iid,
                blocking_merge_request.iid,
                issue_key,
            )
            report.add_created_dependency(
                direction=direction,
                blocked_mr_iid=blocked_mr_iid,
                blocking_mr_iid=blocking_merge_request.iid,
                issue_key=issue_key,
            )
            return

        report.add_existing_dependency(
            direction=direction,
            blocked_mr_iid=blocked_mr_iid,
            blocking_mr_iid=blocking_merge_request.iid,
            issue_key=issue_key,
        )

    async def _get_dependency_map(self, project_id: int, mr_iid: int) -> dict[int, int]:
        cache_key = (project_id, mr_iid)
        if cache_key not in self._dependency_cache:
            self._dependency_cache[cache_key] = await self._gitlab_client.get_dependencies(
                project_id=project_id,
                mr_iid=mr_iid,
            )
        return self._dependency_cache[cache_key]


@lru_cache
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.validate()
    return settings


app = FastAPI()


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def gitlab_webhook(request: Request) -> dict[str, Any]:
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    if settings.gitlab_webhook_secret:
        received_secret = request.headers.get("X-Gitlab-Token")
        if received_secret != settings.gitlab_webhook_secret:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook token",
            )

    payload = await request.json()
    try:
        event = parse_merge_request_event(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if event is None:
        return {"status": "ignored", "reason": "Not a supported merge request event"}

    if not event.jira_key:
        return {
            "status": "skipped",
            "reason": "No Jira key found in MR title",
            "action": event.action,
            "project_id": event.project_id,
            "mr_iid": event.mr_iid,
        }

    logger.info(
        "Processing MR webhook action=%s mr=!%s project_id=%s jira_key=%s",
        event.action,
        event.mr_iid,
        event.project_id,
        event.jira_key,
    )

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        jira_client = JiraClient(http_client, settings)
        gitlab_client = GitLabClient(http_client, settings)
        sync_service = DependencySyncService(jira_client, gitlab_client)

        try:
            report = await sync_service.sync(event)
        except JiraApiError as exc:
            logger.exception("Webhook sync failed during Jira step: %s", exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        except GitLabApiError as exc:
            logger.exception("Webhook sync failed during GitLab step: %s", exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            logger.exception("Webhook sync failed due to network error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Network error while syncing dependency: {exc}",
            ) from exc

    return report.to_dict()
