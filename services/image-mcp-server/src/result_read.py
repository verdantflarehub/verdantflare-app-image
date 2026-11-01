"""Current visibility of a final Image artifact for Studio business reads."""
from __future__ import annotations

import hmac
import os
import re
import uuid

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from .artifacts import ArtifactNotFound
from .channel_authority import authorize
from .content_rating import include_nsfw_from_headers


async def read_result(request, artifacts, ratings, authority_client):
    def respond(body, status=200):
        return JSONResponse(body, status_code=status, headers={"Cache-Control": "no-store"})

    # This is a service-to-service business read, never a query-token endpoint.
    token = os.environ.get("IMAGE_MCP_BEARER_TOKEN", "").strip()
    if not token or not hmac.compare_digest(
        request.headers.get("authorization", ""), "Bearer " + token
    ):
        return respond({"error": "unauthorized"}, 401)
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {"project_id", "artifact_id"}:
            raise ValueError()
        project_id, artifact_id = body["project_id"], body["artifact_id"]
        if not isinstance(project_id, str) or str(uuid.UUID(project_id)) != project_id:
            raise ValueError()
        if not isinstance(artifact_id, str) or not re.fullmatch(r"art-[0-9a-f]{16}", artifact_id):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        return respond({"error": "invalid_argument"}, 400)

    try:
        await run_in_threadpool(authorize, authority_client, project_id, require_write=False)
    except ValueError as exc:
        code = str(exc)
        if code == "verified_studio_identity_required":
            return respond({"error": "unauthorized"}, 401)
        if code == "project_access_denied":
            return respond({"error": "access_denied"}, 403)
        return respond({"error": "result_status_unavailable"}, 503)

    try:
        artifact = await run_in_threadpool(artifacts.get, artifact_id, project_id)
        if artifact.project_id != project_id or artifact.artifact_id != artifact_id:
            return respond({"error": "not_found"}, 404)
        # No task fallback: a faceswap artifact has its own rating identity.
        rating = await run_in_threadpool(ratings.get, artifact_id)
        if rating["rating"] == "nsfw" and not include_nsfw_from_headers(request.headers):
            return respond({"error": "not_found"}, 404)
        return respond({
            "artifact_id": artifact.artifact_id,
            "project_id": artifact.project_id,
            "sha256": artifact.sha256,
            "content_rating": {key: rating[key] for key in ("rating", "source", "human_cleared")},
        })
    except ArtifactNotFound:
        return respond({"error": "not_found"}, 404)
    except Exception:
        # Never turn storage/rating failures into a visible, unrated result.
        return respond({"error": "result_status_unavailable"}, 503)
