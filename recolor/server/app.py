"""The Recolor HTTP API (docs/ARCHITECTURE.md §3.7) and static hosting of `web/`.

Every error is JSON `{"error": <short code>, "detail": <human text>}` with a proper
status code; CORS is open (`X-Render-Ms` exposed). Handlers are plain `def`s so
FastAPI runs them in its thread pool while the worker thread owns the GPU; the SSE
stream is the one `async` handler because it polls a queue with a heartbeat.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
import mimetypes
import os
import queue
import re
import time
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

import recolor
from .. import config, imageio, jobs, pipeline
from ..pipeline import PipelineError
from ..types import Palette, mapping_to_json

MAX_UPLOAD_BYTES = 60 * 1024 * 1024
SSE_HEARTBEAT_S = 15.0
SSE_POLL_S = 0.05
THUMB_MIN_W, THUMB_MAX_W = 16, 2048

LAYER_FILES: dict[str, tuple[str, ...]] = {
    "original": ("original.jpg", "original.png"),
    "work": ("work.png",),
    "preview": ("preview.jpg",),
    "albedo": ("layers/albedo.jpg",),
    "shading": ("layers/shading.jpg",),
    "residual": ("layers/residual.jpg",),
    "regions": ("layers/regions.png",),
    "groups": ("layers/groups.png",),
    "edges": ("layers/edges.png",),
}
ID_FILES = {"regions": "ids/regions.png", "groups": "ids/groups.png"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,199}$")
_PALETTE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ApiError(Exception):
    """Raised by handlers; rendered as `{"error", "detail"}` with `status`."""

    def __init__(self, status: int, error: str, detail: str = ""):
        super().__init__(detail or error)
        self.status, self.error, self.detail = status, error, detail or error


_STATUS_CODES = {400: "bad_request", 404: "not_found", 409: "conflict", 413: "too_large",
                 415: "unsupported_media", 422: "invalid", 500: "internal", 503: "unavailable"}


def _error(status: int, detail: str, error: Optional[str] = None) -> JSONResponse:
    return JSONResponse({"error": error or _STATUS_CODES.get(status, "error"), "detail": detail}, status_code=status)


def _job_or_404(job_id: str) -> jobs.Job:
    job = jobs.registry.get(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", f"no job {job_id!r}")
    return job


def _json_body(request: Request) -> dict[str, Any]:
    """Parse a JSON object body from inside a sync handler (FastAPI runs those on an
    anyio worker thread, so the async body read is bridged with `from_thread.run`)."""
    import anyio
    raw = anyio.from_thread.run(request.body)
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ApiError(400, "bad_json", f"request body is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise ApiError(400, "bad_json", "request body must be a JSON object")
    return data


def _has_dot_segments(request: Request) -> bool:
    """True for paths with a `.` or `..` segment, or a percent-encoded `/` or `.` in
    the raw (undecoded) path - none of the routes accept those."""
    if any(seg in (".", "..") for seg in request.url.path.split("/")):
        return True
    raw = request.scope.get("raw_path") or b""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "surrogateescape")
    raw = raw.lower()
    return b"%2f" in raw or b"%2e" in raw


def _decode_and_create(data: bytes, name: str, options: Any) -> jobs.Job:
    """Decode the upload and create the job record (CPU work: runs off the event loop)."""
    try:
        image = imageio.load_image(data)
    except Exception as e:  # noqa: BLE001 - PIL raises many types for junk input
        raise ApiError(415, "unsupported_image", f"could not decode the image ({type(e).__name__}); "
                                                 "JPEG, PNG, WebP, TIFF and BMP are supported") from None
    if min(image.shape[:2]) < 16:
        raise ApiError(400, "image_too_small", "the image must be at least 16 pixels on each side")
    return jobs.registry.create(image, name, options)


def _sse(event: dict[str, Any], event_id: Optional[int] = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}data: {json.dumps(event, separators=(',', ':'))}\n\n"


def _sample_manifest() -> dict[str, dict[str, Any]]:
    try:
        with open(os.path.join(config.SAMPLES_DIR, "MANIFEST.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _sample_path(name: str) -> str:
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ApiError(404, "sample_not_found", f"no sample {name!r}")
    p = os.path.join(config.SAMPLES_DIR, name)
    if not os.path.isfile(p):
        raise ApiError(404, "sample_not_found", f"no sample {name!r}")
    return p


def _thumb(path: str, width: int, cache_dir: str) -> str:
    """Path of a cached JPEG thumbnail of `path` at `width` (never upscaled)."""
    width = max(THUMB_MIN_W, min(THUMB_MAX_W, width))
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(cache_dir, f"{stem}_w{width}.jpg")
    if os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out
    img = imageio.load_image(path)
    if img.shape[1] > width:
        img = imageio.resize_to(img, (width, max(1, round(img.shape[0] * width / img.shape[1]))))
    tmp = out + ".tmp.jpg"
    imageio.save_image(tmp, img, quality=88)
    os.replace(tmp, out)
    return out


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"device": "cpu", "gpu": None, "vram_total_mb": 0, "vram_used_mb": 0}
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info.update(device="cuda", gpu=torch.cuda.get_device_name(0),
                        vram_total_mb=int(total / 2**20), vram_used_mb=int((total - free) / 2**20))
    except Exception:  # noqa: BLE001 - health must never fail
        pass
    return info


def create_app(warmup: bool = False) -> FastAPI:
    """Build the application. `warmup=True` queues the model warm-up on the GPU
    worker at startup (serve.py does this; tests do not)."""
    app = FastAPI(title="Recolor", version=recolor.__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Render-Ms", "Content-Disposition"])

    # ------------------------------------------------------------------ errors

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return _error(e.status, e.detail, e.error)

    @app.exception_handler(PipelineError)
    async def _pipeline_error(_: Request, e: PipelineError):
        return _error(e.status, e.message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, e: StarletteHTTPException):
        detail = e.detail if isinstance(e.detail, str) else json.dumps(e.detail)
        return _error(e.status_code, detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, e: RequestValidationError):
        parts = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return _error(422, "; ".join(parts) or "invalid request")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        return _error(500, f"{type(e).__name__}: {e}")

    @app.middleware("http")
    async def _no_cache(request: Request, call_next):
        # Dot segments (or their percent-encoded spellings) never name anything we
        # serve; without this an unmatched `/api/palettes/..%2F..` falls through to the
        # StaticFiles mount, which normalizes it to the web root and answers with the
        # SPA index instead of a 404.
        if _has_dot_segments(request):
            return _error(404, "Not Found")
        response = await call_next(request)
        if "cache-control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        config.ensure_dirs()
        pipeline.start_worker()
        if warmup:
            pipeline.start_warmup()
        pipeline.resume_pending()
        yield

    app.router.lifespan_context = _lifespan

    # ------------------------------------------------------------------ health / samples

    @app.get("/api/health")
    def health():
        info = _gpu_info()
        return {"ok": True, **info, "models": pipeline.model_status(), "jobs": len(jobs.registry),
                "queue": pipeline.queue_length(), "version": recolor.__version__}

    @app.get("/api/samples")
    def samples():
        manifest = _sample_manifest()
        out = []
        for name in sorted(os.listdir(config.SAMPLES_DIR)) if os.path.isdir(config.SAMPLES_DIR) else []:
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            entry = manifest.get(name, {})
            size = entry.get("size")
            if not (isinstance(size, list) and len(size) == 2):
                try:
                    size = list(_image_size(os.path.join(config.SAMPLES_DIR, name)))
                except OSError:
                    continue
            out.append({"name": name, "url": f"/api/samples/{name}", "thumb": f"/api/samples/{name}?w=320",
                        "width": int(size[0]), "height": int(size[1]),
                        "title": str(entry.get("label") or str(entry.get("title", "")).removeprefix("File:")),
                        "source_title": str(entry.get("title", "")).removeprefix("File:"),
                        "license": str(entry.get("license", "")), "artist": str(entry.get("artist", ""))})
        return out

    @app.get("/api/samples/{name}")
    def sample(name: str, w: Optional[int] = None):
        path = _sample_path(name)
        headers = {"Cache-Control": "public, max-age=86400"}
        if w:
            return FileResponse(_thumb(path, int(w), os.path.join(config.CACHE_DIR, "samples")),
                                media_type="image/jpeg", headers=headers)
        return FileResponse(path, media_type=mimetypes.guess_type(path)[0] or "image/jpeg", headers=headers)

    # ------------------------------------------------------------------ jobs

    @app.get("/api/jobs")
    def list_jobs():
        return jobs.registry.list()

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request):
        ctype = (request.headers.get("content-type") or "").lower()
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 4096:
            raise ApiError(413, "too_large", f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        fields: dict[str, Any] = {}
        data: Optional[bytes] = None
        name = "upload.jpg"
        if ctype.startswith("multipart/form-data") or ctype.startswith("application/x-www-form-urlencoded"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise ApiError(400, "no_file", "multipart body needs a 'file' part")
            chunks, total = [], 0
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise ApiError(413, "too_large", f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                chunks.append(chunk)
            data = b"".join(chunks)
            name = os.path.basename(upload.filename or "") or "upload.jpg"
            for k in ("detail", "intrinsic", "max_groups", "delta_e"):
                v = form.get(k)
                if isinstance(v, str):
                    fields[k] = v
        else:
            raw = await request.body()
            if len(raw) > 1 << 20:
                raise ApiError(413, "too_large", "JSON body too large")
            try:
                body = json.loads(raw or b"{}")
            except ValueError as e:
                raise ApiError(400, "bad_json", f"request body is not valid JSON: {e}") from None
            if not isinstance(body, dict):
                raise ApiError(400, "bad_json", "request body must be a JSON object")
            fields = body
            sample_name = body.get("sample")
            if not isinstance(sample_name, str) or not sample_name:
                raise ApiError(400, "no_image", "send multipart 'file' or JSON {\"sample\": name}")
            path = _sample_path(sample_name)
            data = await run_in_threadpool(_read_file, path)
            name = sample_name
        if not data:
            raise ApiError(400, "empty_file", "the uploaded file is empty")
        options = pipeline.parse_analysis_options(fields)
        # Decoding (+ downscale to MAX_INGEST_LONG_SIDE) and writing original/preview take
        # 100-250 ms for a large upload; off the loop so SSE streams and other requests
        # keep flowing.
        job = await run_in_threadpool(_decode_and_create, data, name, options)
        pipeline.enqueue(job)
        return job.snapshot()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return _job_or_404(job_id).snapshot()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = _job_or_404(job_id)
        pipeline.invalidate(job)
        jobs.registry.delete(job.id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str, once: int = 0, timeout: float = 0.0):
        """SSE. Replays the current state on connect (status, five stages, groups,
        done/error), then streams live events with a `: keep-alive` comment every
        15 s. `?once=1` closes after the replay; `?timeout=S` closes after S seconds
        without an event (both handy for curl and tests)."""
        job = _job_or_404(job_id)
        q = job.subscribe()   # subscribe before the snapshot so nothing falls in the gap
        replay = job.replay_events()
        idle_limit = timeout if timeout and timeout > 0 else None

        async def stream():
            seq = 0
            try:
                yield ": connected\n\n"
                for ev in replay:
                    seq += 1
                    yield _sse(ev, seq)
                if once:
                    return
                last_beat = last_event = time.monotonic()
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        ev = q.get_nowait()
                    except queue.Empty:
                        now = time.monotonic()
                        if idle_limit is not None and now - last_event >= idle_limit:
                            return
                        if now - last_beat >= SSE_HEARTBEAT_S:
                            last_beat = now
                            yield ": keep-alive\n\n"
                        await asyncio.sleep(SSE_POLL_S)
                        continue
                    seq += 1
                    last_beat = last_event = time.monotonic()
                    yield _sse(ev, seq)
                    if ev.get("type") == "status" and ev.get("status") == "deleted":
                        return
            finally:
                job.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    @app.get("/api/jobs/{job_id}/layers/{layer}")
    def job_layer(job_id: str, layer: str):
        job = _job_or_404(job_id)
        if layer not in LAYER_FILES:
            raise ApiError(404, "unknown_layer", f"layer must be one of {sorted(LAYER_FILES)}")
        for rel in LAYER_FILES[layer]:
            p = job.path(*rel.split("/"))
            if os.path.isfile(p):
                return FileResponse(p, media_type=mimetypes.guess_type(p)[0] or "application/octet-stream",
                                    headers={"Cache-Control": "no-cache"})
        raise ApiError(404, "layer_not_ready", f"layer {layer!r} is not available yet (job is {job.status})")

    @app.get("/api/jobs/{job_id}/ids/{kind}")
    def job_ids(job_id: str, kind: str):
        job = _job_or_404(job_id)
        if kind not in ID_FILES:
            raise ApiError(404, "unknown_ids", "kind must be 'regions' or 'groups'")
        p = job.path(*ID_FILES[kind].split("/"))
        if not os.path.isfile(p):
            raise ApiError(404, "ids_not_ready", f"{kind} ids are not available yet (job is {job.status})")
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ group edits

    def _edit(job_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = _job_or_404(job_id)
        pipeline.apply_group_edit(job, kind, payload)
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/groups/merge")
    def groups_merge(job_id: str, request: Request):
        return _edit(job_id, "merge", _json_body(request))

    @app.post("/api/jobs/{job_id}/groups/split")
    def groups_split(job_id: str, request: Request):
        return _edit(job_id, "split", _json_body(request))

    @app.post("/api/jobs/{job_id}/groups/move")
    def groups_move(job_id: str, request: Request):
        return _edit(job_id, "move", _json_body(request))

    @app.patch("/api/jobs/{job_id}/groups/{gid}")
    def groups_update(job_id: str, gid: int, request: Request):
        body = _json_body(request)
        allowed = {k: body[k] for k in ("name", "locked", "is_background") if k in body}
        if not allowed:
            raise ApiError(400, "nothing_to_update", "send at least one of name, locked, is_background")
        return _edit(job_id, "update", {"group_id": gid, **allowed})

    @app.post("/api/jobs/{job_id}/regroup")
    def regroup(job_id: str, request: Request):
        return _edit(job_id, "regroup", _json_body(request))

    # ------------------------------------------------------------------ palettes

    def _palette_dir(pid: str) -> str:
        if not _PALETTE_ID.match(pid):
            raise ApiError(404, "palette_not_found", f"no palette {pid!r}")
        d = os.path.join(config.PALETTE_CACHE_DIR, pid)
        if not os.path.isdir(d):
            raise ApiError(404, "palette_not_found", f"no palette {pid!r}")
        return d

    @app.post("/api/palettes")
    def create_palette(request: Request):
        body = _json_body(request)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise ApiError(400, "no_prompt", "prompt must not be empty")
        if len(prompt) > 200:
            raise ApiError(400, "prompt_too_long", "prompt must be at most 200 characters")
        n = body.get("n_colors", 6)
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not (1 <= int(n) <= 16):
            raise ApiError(400, "bad_n_colors", "n_colors must be an integer between 1 and 16")
        from recolor.palette import generate_palette
        pal = generate_palette(prompt, n_colors=int(n))
        return pal.to_dict() if isinstance(pal, Palette) else pal

    @app.get("/api/palettes/{pid}")
    def get_palette(pid: str):
        d = _palette_dir(pid)
        try:
            with open(os.path.join(d, "palette.json"), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            raise ApiError(404, "palette_not_found", f"no palette {pid!r}") from None

    @app.get("/api/palettes/{pid}/sources/{i}.jpg")
    def palette_source(pid: str, i: int):
        d = _palette_dir(pid)
        p = os.path.join(d, "sources", f"{i}.jpg")
        if not os.path.isfile(p):
            raise ApiError(404, "source_not_found", f"palette {pid} has no source {i}")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    # ------------------------------------------------------------------ mapping / render / export

    @app.post("/api/jobs/{job_id}/mapping/suggest")
    def mapping_suggest(job_id: str, request: Request):
        job = _job_or_404(job_id)
        if job.status != "ready":
            raise ApiError(409, "not_ready", f"job is {job.status}, not ready")
        body = _json_body(request)
        colors = body.get("colors")
        if not isinstance(colors, list) or not colors:
            raise ApiError(400, "no_colors", "colors must be a non-empty list of hex strings")
        hexes = []
        for c in colors:
            if isinstance(c, dict) and isinstance(c.get("hex"), str):
                c = c["hex"]
            try:
                hexes.append(imageio.rgb01_to_hex(imageio.hex_to_rgb01(str(c))))
            except ValueError:
                raise ApiError(400, "bad_color", f"{c!r} is not a hex color") from None
        from recolor.mapping import STRATEGIES, suggest_mapping
        strategy = body.get("strategy") or "balanced"
        if strategy not in STRATEGIES:
            raise ApiError(400, "bad_strategy", f"strategy must be one of {list(STRATEGIES)}")
        keep_background = bool(body.get("keep_background", True))
        keep_locked = bool(body.get("keep_locked", True))
        m = suggest_mapping(job.groups(), hexes, strategy=strategy, keep_background=keep_background,
                            keep_locked=keep_locked)
        return {"mapping": mapping_to_json(m), "strategy": strategy}

    @app.post("/api/jobs/{job_id}/render")
    def render(job_id: str, request: Request):
        job = _job_or_404(job_id)
        body = _json_body(request)
        t0 = time.perf_counter()
        data = pipeline.render_preview(job, body.get("mapping"), body.get("options"))
        ms = int((time.perf_counter() - t0) * 1000)
        return Response(content=data, media_type="image/jpeg",
                        headers={"X-Render-Ms": str(ms), "Cache-Control": "no-store"})

    @app.post("/api/jobs/{job_id}/export")
    def export(job_id: str, request: Request):
        job = _job_or_404(job_id)
        body = _json_body(request)
        quality = body.get("quality") or "work"
        fmt = (body.get("format") or "png").lower().replace("jpeg", "jpg")
        res = pipeline.export(job, body.get("mapping"), body.get("options"), quality, fmt)
        return {"url": f"/api/jobs/{job.id}/exports/{res['file']}", "file": res["file"],
                "width": res["width"], "height": res["height"], "ms": res["ms"],
                "intrinsic": res.get("intrinsic")}

    @app.get("/api/jobs/{job_id}/exports/{file}")
    def get_export(job_id: str, file: str):
        job = _job_or_404(job_id)
        if not _SAFE_NAME.match(file) or "/" in file or ".." in file:
            raise ApiError(404, "export_not_found", f"no export {file!r}")
        p = job.path("exports", file)
        if not os.path.isfile(p):
            raise ApiError(404, "export_not_found", f"no export {file!r}")
        media = "image/png" if file.lower().endswith(".png") else "image/jpeg"
        return FileResponse(p, media_type=media, filename=file, content_disposition_type="attachment")

    @app.put("/api/jobs/{job_id}/state")
    def put_state(job_id: str, request: Request):
        job = _job_or_404(job_id)
        body = _json_body(request)
        fields = pipeline.validate_state(job, body)
        if not fields:
            raise ApiError(400, "nothing_to_save", "send at least one of mapping, render_options, palette_id")
        job.update(**fields)
        return job.snapshot()

    # ------------------------------------------------------------------ static web/

    index_path = os.path.join(config.WEB_DIR, "index.html")

    @app.get("/", include_in_schema=False)
    def index():
        if os.path.isfile(index_path):
            return FileResponse(index_path, media_type="text/html", headers={"Cache-Control": "no-store"})
        return HTMLResponse("<!doctype html><meta charset=utf-8><title>Recolor</title>"
                            "<p style='font:16px system-ui;padding:2rem'>The API is up. The web UI "
                            "(<code>web/index.html</code>) is not built yet. See <a href='/api/docs'>/api/docs</a>.")

    if os.path.isdir(config.WEB_DIR):
        app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")

    return app


app = create_app()
