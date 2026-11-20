"""Project-scoped Flow calls and restart-safe publication of generated image bytes."""
from decimal import Decimal, InvalidOperation
import hashlib
import io
import os
import re
from uuid import UUID

import httpx
from PIL import Image
from .channel import ImageChannel, ChannelUnavailable
from .comfyui import new_uuid7

MODELS = ("nano2", "nano21", "nano-pro")
RUNTIME = "google-flow-channel-v1"
MAX_BYTES = 48 << 20


def normalize(params):
    model = params.get("model") or MODELS[0]
    aspect = params.get("aspect_ratio") or "1:1"
    if model not in MODELS or aspect not in ("16:9", "9:16", "1:1"):
        raise ValueError("capability_not_supported")
    if not isinstance(params.get("prompt"), str) or not 0 < len(params["prompt"].strip()) <= 5000:
        raise ValueError("invalid_prompt")
    allowed = {"model", "prompt", "aspect_ratio", "action", "budget_limit", "reference_images",
               "reference_artifact_ids", "source_artifact_id", "resolution", "quality", "background"}
    if not params.get("size"):
        params = {k: v for k, v in params.items() if k != "size"}
    if any(key not in allowed and val is not None for key,val in params.items()):
        raise ValueError("unsupported_parameter")
    if params.get("resolution") or params.get("quality") or params.get("background") not in (None,"auto"):
        raise ValueError("unsupported_parameter")
    if params.get("action") not in ("generate","edit"):
        raise ValueError("unsupported_parameter")
    refs = [params["source_artifact_id"]] if params.get("source_artifact_id") else []
    for key in ("reference_images", "reference_artifact_ids"):
        values = params.get(key) or []
        if not isinstance(values,list):
            raise ValueError("invalid_reference")
        refs.extend(values)
    if len(refs)>3 or any(not isinstance(ref,str) or not re.fullmatch(r"art-[0-9a-f]{16}", ref) for ref in refs):
        raise ValueError("invalid_reference")
    result = dict(model=model, prompt=params["prompt"], aspect_ratio=aspect, action=params["action"],
                  reference_artifact_ids=refs)
    if params.get("budget_limit") is not None:
        try:
            budget = Decimal(str(params["budget_limit"]))
        except InvalidOperation:
            raise ValueError("invalid_budget_limit") from None
        if not budget.is_finite() or not 0 <= budget <= 2**31-1 or budget != budget.to_integral_value():
            raise ValueError("invalid_budget_limit")
        result["budget_limit"] = int(budget)
    return result


def image_info(data, mime, digest):
    if not 0 < len(data) <= MAX_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("image_integrity_failed")
    with Image.open(io.BytesIO(data)) as image:
        if mime not in {"image/png","image/jpeg","image/webp"} or Image.MIME.get(image.format) != mime or image.width*image.height>40_000_000:
            raise ValueError("invalid_image")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.size


class FlowChannel(ImageChannel):
    def headers(self, project, actor):
        return {"Authorization":"Bearer "+self.token,"X-User-Id":actor["user_id"],
                "X-Organization-Id":actor["organization_id"],"X-Project-Id":project,"X-Project-Write":"true"}

    def check(self, project, actor, params):
        headers = self.headers(project,actor)
        try:
            models = self.client.get(self.endpoint + "/v1/models", headers=headers).raise_for_status().json()
        except (httpx.HTTPError, ValueError):
            raise ValueError("channel_discovery_unavailable") from None
        if params["model"] not in models["models"]["image"]:
            raise ValueError("capability_not_supported")

    def execute(self, manager, job, recovery=False):
        tasks = manager.tasks
        with manager._task_lock(job.task_id):
            rec = tasks.get(job.task_id)
            if rec.status != "running":
                return
            saved = job.params["channel_authority"]
            actor = manager.google_authorize(job.project_id,saved)
            if actor != saved:
                raise ValueError("channel_authority_mismatch")
            headers = self.headers(job.project_id,actor)
            if recovery or rec.phase:
                response = self.client.get(self.endpoint+"/v1/calls/by-idempotency/"+job.task_id,headers=headers)
            else:
                args = {"prompt":job.params["prompt"],"aspect":job.params["aspect_ratio"]}
                refs = []
                for ref in job.params["reference_artifact_ids"]:
                    asset = manager.artifacts.get(ref,job.project_id)
                    raw = manager.artifacts.content_path(asset).read_bytes()
                    if len(raw) != asset.size_bytes or asset.project_id != job.project_id:
                        raise ValueError("invalid_reference")
                    image_info(raw,asset.media_type,asset.sha256)
                    receipt = self.client.put(self.endpoint+"/v1/inputs", headers=headers | {
                        "Content-Type":asset.media_type, "X-Content-SHA256":asset.sha256},content=raw).raise_for_status().json()
                    if receipt.get("sha256") != asset.sha256 or receipt.get("size") != len(raw):
                        raise ValueError("input_receipt_mismatch")
                    refs.append(asset.sha256)
                if refs:
                    args["reference_images"] = refs
                payload = dict(idempotency_key=job.task_id,model=job.params["model"],capability="image",parameters=args)
                if "budget_limit" in job.params:
                    payload["budget_limit"] = job.params["budget_limit"]
                if not tasks.flow_intent(job.task_id):
                    return
                response = self.client.post(self.endpoint+"/v1/calls",headers=headers,json=payload)
            result = response.raise_for_status().json()
            cid = result.get("call_id")
            if result.get("request_id") != job.task_id or not isinstance(cid,str) or not re.fullmatch("[a-f0-9]{32}",cid):
                raise ChannelUnavailable("invalid_channel_response")
            tasks.flow_call(job.task_id, cid)
            if result["state"] in ("failed","rejected"):
                tasks.update_status(job.task_id,"failed",phase="failed",error="channel_rejected")
                return
            if result.get("accepted") is not True or result["state"] != "succeeded":
                phase = "submission_unknown" if result.get("accepted") is None else "channel_running"
                tasks.update_status(job.task_id,"running",phase=phase)
                return
            tasks.update_status(job.task_id,"running",phase="collecting")
            with self.client.stream("GET",self.endpoint+"/v1/calls/"+cid+"/result",headers=headers) as resp:
                resp.raise_for_status()
                mime = resp.headers.get("content-type","").split(";")[0]
                digest = resp.headers.get("X-Content-SHA256")
                raw = bytearray()
                for chunk in resp.iter_bytes():
                    if len(raw)+len(chunk)>MAX_BYTES:
                        raise ValueError("invalid_image")
                    raw.extend(chunk)
            data = bytes(raw)
            width,height = image_info(data,mime,digest)
            a,b = map(int,job.params["aspect_ratio"].split(":"))
            if abs((width / height) - (a / b)) / (a / b) > 0.05:
                raise ValueError("image_aspect_mismatch")
            ref = self.publish(tasks,job,cid,actor,data,mime,digest)
            artifact = manager.artifacts.create_from_bytes(project_id=job.project_id,
                artifact_id="art-"+hashlib.sha256(job.task_id.encode()).hexdigest()[:16],
                filename=job.task_id+"."+mime.split("/")[1],data=data,media_type=mime,
                metadata={"engine":"google-flow","channel_call_id":cid,"content_ref":ref,"task_id":job.task_id,
                          "width":width,"height":height,"credits":result.get("credits")})
            tasks.update_status(job.task_id,"completed",artifact_id=artifact.artifact_id,phase="completed",error="")

    def publish(self,tasks,job,cid,actor,data,mime,digest):
        url = os.environ.get("IMAGE_ARTIFACT_SERVICE_URL","").rstrip("/")
        token = os.environ.get("IMAGE_ARTIFACT_SERVICE_TOKEN","")
        if not url or not token:
            raise ValueError("artifact_service_unavailable")
        headers = {"Authorization":"Bearer "+token,"X-User-Id":actor["user_id"],"X-Organization-Id":actor["organization_id"]}
        def request(method,path,**kwargs):
            return self.client.request(method,url+path,headers=headers | {"X-Request-Id":new_uuid7()} |
                kwargs.pop("headers",{}),follow_redirects=False,**kwargs).raise_for_status()
        source = dict(kind="task_output",service_id="image",project_id=job.project_id,run_id=job.task_id,task_id=str(UUID(cid)))
        write_id, upload_id = tasks.flow_upload(job.task_id,cid,digest)
        if not upload_id:
            receipt = request("POST","/v2/artifacts/uploads",json=dict(write_id=write_id,source=source,
                              sha256=digest,size=len(data),mime=mime)).json()
            upload_id = str(UUID(receipt["upload_id"]))
            tasks.flow_upload(job.task_id,cid,digest,upload_id)
        path = "/v2/artifacts/uploads/"+upload_id
        state = request("GET",path).json()["state"]
        if state == "prepared":
            request("PUT",path+"/content",content=data,headers={"Content-Type":mime})
        elif state != "committed":
            raise ValueError("invalid_upload_state")
        version = request("POST",path+"/commit").json()
        expected = dict(source=source,sha256=digest,size=len(data),mime=mime,
                        organization_id=actor["organization_id"],created_by=actor["user_id"])
        if any(version.get(k)!=v for k,v in expected.items()):
            raise ValueError("artifact_receipt_mismatch")
        ref = {k:str(UUID(version[k])) for k in ("artifact_id","store_id","version_id")}
        path = url + "/v2/artifacts/" + ref["version_id"] + "/content"
        with self.client.stream("GET", path, params={"store_id": ref["store_id"], "artifact_id": ref["artifact_id"]},
                                headers=headers | {"X-Request-Id": new_uuid7()}, follow_redirects=False) as response:
            response.raise_for_status()
            offset = 0
            for chunk in response.iter_bytes(chunk_size=65536):
                if data[offset:offset + len(chunk)] != chunk:
                    raise ValueError("artifact_readback_mismatch")
                offset += len(chunk)
            if offset != len(data):
                raise ValueError("artifact_readback_mismatch")
        return ref
