"""Analysis orchestration, the single GPU worker, model warm-up, renders and exports.

The sibling modules (`recolor.intrinsic`, `recolor.segmentation.*`, `recolor.engine`,
`recolor.palette`, `recolor.mapping`) are imported lazily by name inside the functions
that need them, exactly against the signatures in docs/ARCHITECTURE.md §3, so this
module imports cleanly while they are still being written and tests can substitute
them through `sys.modules`.

Concurrency model:
- One daemon worker thread drains a FIFO queue of analysis jobs (the GPU is shared).
- Model warm-up is the first task on that same queue, so it can never race a job for
  the lazy model singletons; `model_status()` reports cold/loading/ready for the UI.
- Exports take `gpu_lock`, which the worker also holds per stage, so a full-resolution
  export slots in between stages instead of colliding with a 14 GB intrinsic pass.
- Preview renders are serialized by a separate lock and never wait for the worker.

Model lifecycle: SAM 2, the intrinsic model, ViTMatte (the groups stage's boundary snap),
Florence-2 (lettering and named parts) and BiRefNet (the foreground matte) are lazy
singletons loaded on first use (analysis, or a full-resolution export). A background
watchdog (started by `start_worker`, alongside the worker thread) drops them again after
`config.IDLE_UNLOAD_S` seconds with no analysis or full-resolution export running, so the
GPU holds their memory only while the app is actually being used. A startup `--warmup`
just moves the first load earlier; it does not exempt the models from being unloaded
later. Preview renders and working-resolution exports never touch any model, so tuning
colors on an already-analyzed job never keeps them resident.

"Ignore background" (`ignore_background` in the job record, on for new jobs): every group
the analysis flagged as background is treated as locked by the renderer, the exports and the
mapping suggestions (`effective_groups`); the stored flags are not touched, so the user's
own locks and the regroup keep working as before.

Per-job masks: the groups stage also writes `islands.npy` (decals carved out of the paint)
and `protect.npy` (own-coloured objects the engine's reflection stage must not recolour),
bool at working resolution, and `regroup.npz` (the regions stage's label map and every final
region's origin in it) so that a regroup reproduces the analysis. Jobs analysed before they
existed have none of these files and render with no islands and no protected pixels; their
group edits keep the old behaviour.

Ownership: a queued or analysing job names its server process in `owner.json`; a server
starting on the same data directory resumes only jobs whose owner is gone, and leaves
unclaimed jobs alone while one of them is still making progress (a server that predates
owner files is working through them; see `RESUME_ACTIVE_S`).

Group edits of one job are serialised by `Job.edit_lock`; renders and exports read the
layers and groups of one edit generation (`_snapshot`), never a half-written edit.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import queue
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import replace
from typing import Any, Callable, Iterable, Optional

import numpy as np

from . import config, filters, imageio
from .jobs import Job, registry
from .types import STAGES, AnalysisOptions, ColorGroup, Mapping, Region, RenderOptions, mapping_to_json

log = logging.getLogger("recolor.pipeline")

DETAIL_LEVELS = ("fast", "balanced", "max")
INTRINSIC_METHODS = ("auto", "careaga", "heuristic")
GROUP_EDIT_KINDS = ("merge", "split", "move", "regroup", "update")
SPLIT_MODES = ("colour", "instances")     # split by albedo k-means, or a part group into its instances
NAME_MAX_LEN = 48                      # a group name from PATCH /groups/{gid}, after trimming
MAX_GROUPS_LIMIT = 256                 # max_groups of a job and of a regroup: 1..256 (or null: auto)
DELTA_E_MIN, DELTA_E_MAX = 0.5, 100.0  # delta_e of a job and of a regroup
EXPORT_QUALITIES = ("work", "full")
EXPORT_FORMATS = ("png", "jpg")

# Guided label refinement allocates several [G,H,W] float tensors; above this many
# elements per tensor the export falls back to a plain nearest upsample rather than
# risking a CUDA OOM on a 24-megapixel original with many groups.
_REFINE_MAX_ELEMENTS = 320_000_000

gpu_lock = threading.RLock()
_render_lock = threading.Lock()


class PipelineError(Exception):
    """A user-facing failure with an HTTP status: bad input (400), missing artifact
    (404), a job in the wrong state (409) or a GPU too full to finish (503, with a
    ``retry_after`` hint in seconds). Anything else is a real bug."""

    def __init__(self, status: int, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


# =============================================================================
# lazy sibling imports
# =============================================================================

def _mod(name: str):
    return importlib.import_module(name)


def _intrinsic():
    return _mod("recolor.intrinsic")


def _sam_masks():
    return _mod("recolor.segmentation.sam_masks")


def _hierarchy():
    return _mod("recolor.segmentation.hierarchy")


def _grouping():
    return _mod("recolor.segmentation.grouping")


def _engine():
    return _mod("recolor.engine")


def _refine():
    return _mod("recolor.segmentation.refine")


def _matting():
    return _mod("recolor.segmentation.matting")


def _florence():
    return _mod("recolor.segmentation.florence")


def _foreground():
    return _mod("recolor.segmentation.foreground")


def _smallparts():
    return _mod("recolor.segmentation.smallparts")


def _partdetect():
    return _mod("recolor.segmentation.partdetect")


def _junk():
    return _mod("recolor.segmentation.junk")


def _subject():
    return _mod("recolor.segmentation.subject")


# =============================================================================
# worker thread and warm-up
# =============================================================================

_task_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_current_job_id: Optional[str] = None

_models: dict[str, str] = {"sam2": "cold", "intrinsic": "cold", "vitmatte": "cold", "florence": "cold",
                           "birefnet": "cold", "owlv2": "cold"}
#: The lazy segmentation singletons besides SAM 2 and ViTMatte, by their `model_status` name.
_AUX_MODELS = (("florence", "_florence"), ("birefnet", "_foreground"), ("owlv2", "_partdetect"))


def _aux_models():
    """``[(name, getter)]`` of Florence-2, BiRefNet and OWLv2 (looked up at call time, so a
    test can replace a getter)."""
    return [(name, globals()[getter]) for name, getter in _AUX_MODELS]
_warmup_started = False

# Idle-unload: SAM 2, the intrinsic model and ViTMatte are dropped from the GPU after
# `config.IDLE_UNLOAD_S` seconds with no analysis or full-resolution export running,
# then reloaded lazily (like any other lazy singleton) on the next one.
_last_activity: float = time.monotonic()
_idle_watchdog: Optional[threading.Thread] = None
_idle_watchdog_lock = threading.Lock()
_IDLE_CHECK_PERIOD_S = 5.0


def _touch_activity() -> None:
    """Record that the GPU models were just asked to do something, resetting the
    idle-unload countdown."""
    global _last_activity
    _last_activity = time.monotonic()


def start_worker() -> None:
    """Start the single FIFO GPU worker and the idle-unload watchdog (idempotent)."""
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, name="recolor-worker", daemon=True)
            _worker.start()
    _start_idle_watchdog()


def _start_idle_watchdog() -> None:
    global _idle_watchdog
    with _idle_watchdog_lock:
        if _idle_watchdog is not None and _idle_watchdog.is_alive():
            return
        _idle_watchdog = threading.Thread(target=_idle_watchdog_loop, name="recolor-idle-unload", daemon=True)
        _idle_watchdog.start()


def _idle_watchdog_loop() -> None:
    while True:
        time.sleep(_IDLE_CHECK_PERIOD_S)
        _idle_tick()


def _idle_tick() -> None:
    """One idle-unload check. Split out from the sleep loop so it can be called
    directly (and its timing/locking stubbed) without waiting on a real thread."""
    limit = config.IDLE_UNLOAD_S
    if limit <= 0 or time.monotonic() - _last_activity < limit:
        return
    if _current_job_id is not None or queue_length() > 0:
        return
    # A running stage or a full-resolution export holds gpu_lock for as long as it
    # needs the model; failing to acquire it here just means something is using the
    # GPU right now, so this is the only guard needed against unloading mid-use.
    if not gpu_lock.acquire(blocking=False):
        return
    try:
        _release_idle_models()
    finally:
        gpu_lock.release()


def _release_segmentation_models(why: str = "idle-unload") -> list[str]:
    """Drop SAM 2, ViTMatte, Florence-2, BiRefNet and OWLv2 if loaded (they reload lazily on
    the next job that needs them) and return the names released. Called only with `gpu_lock`
    held. Errors are logged, not raised: a failed release just leaves the model resident."""
    released: list[str] = []
    try:
        sam = _sam_masks()
        if callable(getattr(sam, "is_loaded", None)) and sam.is_loaded() and callable(getattr(sam, "release", None)):
            sam.release()
            released.append("sam2")
    except Exception:  # noqa: BLE001
        log.exception("%s: failed to release SAM 2", why)
    try:
        matting = _matting()
        if callable(getattr(matting, "is_loaded", None)) and matting.is_loaded():
            matting.release()
            released.append("vitmatte")
    except Exception:  # noqa: BLE001
        log.exception("%s: failed to release ViTMatte", why)
    for name, get in _aux_models():
        try:
            mod = get()
            if callable(getattr(mod, "is_loaded", None)) and mod.is_loaded():
                mod.release()
                released.append(name)
        except Exception:  # noqa: BLE001
            log.exception("%s: failed to release %s", why, name)
    for name in released:
        _models[name] = "cold"
    return released


def _release_idle_models() -> None:
    """Drop SAM 2, the intrinsic model, ViTMatte, Florence-2, BiRefNet and OWLv2 if loaded,
    freeing the CUDA memory they hold. Called only with `gpu_lock` held. Errors are logged,
    not raised: a failed release just leaves the model resident until the next check."""
    released = []
    try:
        if _intrinsic().is_loaded("careaga"):
            _intrinsic().release("careaga")
            released.append("intrinsic")
    except Exception:  # noqa: BLE001
        log.exception("idle-unload: failed to release the intrinsic model")
    released += _release_segmentation_models()
    if released:
        if "intrinsic" in released:
            _models["intrinsic"] = "cold"
        log.info("released %s after %.0f s idle", " + ".join(released), config.IDLE_UNLOAD_S)


def _worker_loop() -> None:
    global _current_job_id
    while True:
        kind, payload = _task_queue.get()
        try:
            if kind == "warmup":
                _warmup_task()
            elif kind == "job":
                job = registry.get(payload)
                if job is not None:
                    _current_job_id = job.id
                    analyze(job)
        except Exception:  # noqa: BLE001 - the worker must survive anything
            log.exception("worker task %s failed", kind)
        finally:
            _current_job_id = None
            _task_queue.task_done()


def enqueue(job: Job) -> None:
    """Queue a job for analysis (FIFO) and make sure the worker is running. The job is
    marked as owned by this server process (`OWNER_FILE`) until its analysis ends, so a
    second server on the same data directory does not take it over."""
    start_worker()
    _claim(job)
    _task_queue.put(("job", job.id))


# A job queued or being analysed records which server process owns it. `resume_pending`
# only re-queues jobs whose owner is gone: a second server started on the same data
# directory used to reset and fail a job another live server was analysing.
OWNER_FILE = "owner.json"


def _process_start(pid: int) -> Optional[str]:
    """The kernel's start time of process ``pid`` (Linux /proc), which tells a live owner
    from an unrelated process that reused its pid; None when it cannot be read."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as f:
            stat = f.read().decode("ascii", "replace")
        return stat[stat.rindex(")") + 2:].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _claim(job: Job) -> None:
    import socket
    owner = {"pid": os.getpid(), "host": socket.gethostname(), "start": _process_start(os.getpid()),
             "claimed": time.time()}
    try:
        with open(job.path(OWNER_FILE), "w", encoding="utf-8") as f:
            json.dump(owner, f)
    except OSError as e:
        log.warning("job %s: could not record its owner (%s)", job.id, e)


def _release_claim(job: Job) -> None:
    try:
        os.remove(job.path(OWNER_FILE))
    except OSError:
        pass


def _owned_elsewhere(job: Job) -> bool:
    """True when another live server process on this machine owns the job."""
    import socket
    try:
        with open(job.path(OWNER_FILE), "r", encoding="utf-8") as f:
            owner = json.load(f)
        pid = int(owner["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if pid == os.getpid() or owner.get("host") != socket.gethostname():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                                  # alive, owned by another user
    except OSError:
        return False
    start = _process_start(pid)
    return start is None or owner.get("start") in (None, start)


def queue_length() -> int:
    """Jobs waiting in the FIFO (not counting the one being analyzed)."""
    return _task_queue.qsize()


# A server started from code older than owner files claims nothing: a job it is analysing
# looks abandoned. It still rewrites job.json at every stage transition (the longest stage,
# SAM at Max, takes up to ~40 s on the shared card), so an unclaimed queued / analysing job
# whose job.json changed within RESUME_ACTIVE_S means such a server is alive and working
# through its queue: every unclaimed job is left to it and looked at again later
# (`_recheck_deferred`), and resumed only once they have all been quiet that long.
RESUME_ACTIVE_S = 90.0
_RECHECK_EVERY_S = 30.0


def _quiet_for(job: Job) -> float:
    """Seconds since job.json was last written (infinite when it cannot be read)."""
    try:
        return max(0.0, time.time() - os.path.getmtime(job.path("job.json")))
    except OSError:
        return float("inf")


def _reset_and_enqueue(job: Job) -> None:
    with job.lock:
        for s in STAGES:
            job.meta["stages"][s] = {"state": "idle", "progress": 0.0, "message": "", "seconds": 0.0}
        job.meta["status"] = "queued"
        job.meta["error"] = None
    job.save()
    enqueue(job)


def resume_pending() -> int:
    """Re-queue jobs that were `queued`/`analyzing` when the server last stopped.
    Their stages are reset; the original image on disk is enough to restart. A job that
    another live server process still owns (`OWNER_FILE`) is left alone, and so are all
    unclaimed ones while one of them shows recent progress (a server without owner files
    is at work, see RESUME_ACTIVE_S); those are re-checked in the background. Returns the
    number of jobs queued now."""
    unclaimed: list[Job] = []
    stale_claims: list[Job] = []
    for job in reversed(registry.all()):
        if job.status not in ("queued", "analyzing"):
            continue
        if _owned_elsewhere(job):
            log.info("job %s is being analysed by another server process; not resuming it", job.id)
            continue
        (stale_claims if os.path.exists(job.path(OWNER_FILE)) else unclaimed).append(job)
    now: list[Job] = list(stale_claims)
    if unclaimed and min(_quiet_for(j) for j in unclaimed) < RESUME_ACTIVE_S:
        log.info("jobs %s show recent progress without an owner (a server predating owner files); "
                 "leaving them to it and checking again every %.0f s", [j.id for j in unclaimed], _RECHECK_EVERY_S)
        _start_deferred_resume(unclaimed)
    else:
        now += unclaimed
    for job in now:
        _reset_and_enqueue(job)
    return len(now)


def _recheck_deferred(jobs: list[Job]) -> list[Job]:
    """One look at jobs left to another server: a job it finished (on disk) takes the
    record from disk and publishes it; when none of the rest has progressed for
    RESUME_ACTIVE_S, they are all resumed here. Returns the jobs still left to the other
    server."""
    left: list[Job] = []
    quiet = float("inf")                        # a job the other server just finished counts too
    for job in jobs:
        if not _job_alive(job):
            continue
        quiet = min(quiet, _quiet_for(job))
        try:
            disk = Job.load(job.dir).meta
        except (OSError, ValueError):
            continue
        if disk.get("status") not in ("queued", "analyzing"):
            with job.lock:
                job.meta = disk
            for event in job.replay_events():
                job.publish(event)
            continue
        if _owned_elsewhere(job):
            continue                            # claimed by a newer server meanwhile
        left.append(job)
    if left and quiet >= RESUME_ACTIVE_S:
        log.info("jobs %s made no progress for %.0f s; resuming them", [j.id for j in left], RESUME_ACTIVE_S)
        for job in left:
            _reset_and_enqueue(job)
        return []
    return left


def _start_deferred_resume(jobs: list[Job]) -> None:
    threading.Thread(target=_deferred_resume_loop, args=(jobs,), name="recolor-resume", daemon=True).start()


def _deferred_resume_loop(jobs: list[Job]) -> None:
    while jobs:
        time.sleep(_RECHECK_EVERY_S)
        try:
            jobs = _recheck_deferred(jobs)
        except Exception:  # noqa: BLE001 - never kill the server over this
            log.exception("re-checking deferred jobs failed")
            return


def start_warmup() -> None:
    """Load SAM 2 and the intrinsic models on the worker thread so the first job is
    fast. Idempotent; failures are logged and leave the models `cold` (the pipeline
    still loads them lazily on first use)."""
    global _warmup_started
    with _worker_lock:
        if _warmup_started:
            return
        _warmup_started = True
    start_worker()
    _task_queue.put(("warmup", None))


def _warmup_task() -> None:
    _touch_activity()
    _models["intrinsic"] = "loading"
    try:
        with gpu_lock:
            _intrinsic().warmup("careaga")
        _models["intrinsic"] = "ready"
    except Exception as e:  # noqa: BLE001
        log.warning("intrinsic warm-up failed: %s", e)
        _models["intrinsic"] = "cold"
    _models["sam2"] = "loading"
    try:
        with gpu_lock:
            sam = _sam_masks()
            if callable(getattr(sam, "warmup", None)):
                sam.warmup()
            else:
                masker = sam.SamMasker()
                loader = next((getattr(masker, n) for n in ("warmup", "load", "ensure_loaded")
                               if callable(getattr(masker, n, None))), None)
                if loader is not None:
                    loader()
                else:
                    # No explicit loader: a tiny generate loads the model and compiles
                    # the kernels, which is exactly what warm-up is for.
                    probe = np.full((64, 64, 3), 128, np.uint8)
                    probe[16:48, 16:48] = (200, 40, 40)
                    masker.generate(probe, detail="fast")
        _models["sam2"] = "ready"
    except Exception as e:  # noqa: BLE001
        log.warning("SAM 2 warm-up failed: %s", e)
        _models["sam2"] = "cold"
    _models["vitmatte"] = "loading"
    try:
        with gpu_lock:
            matting = _matting()
            matting.warmup()
            _models["vitmatte"] = matting.status()
    except Exception as e:  # noqa: BLE001
        log.warning("ViTMatte warm-up failed: %s", e)
        _models["vitmatte"] = "cold"
    for name, get in _aux_models():
        _models[name] = "loading"
        try:
            with gpu_lock:
                mod = get()
                mod.warmup()
                _models[name] = mod.status()
        except Exception as e:  # noqa: BLE001
            log.warning("%s warm-up failed: %s", name, e)
            _models[name] = "cold"
    _free_cuda()


def model_status() -> dict[str, str]:
    """`{sam2, intrinsic, vitmatte, florence, birefnet, owlv2}` each 'cold' | 'loading' |
    'ready' ('unavailable' for ViTMatte, Florence-2, BiRefNet and OWLv2 when their package or
    weights are missing: the analysis then runs without that piece). Consults the siblings'
    probes when importable, so a model loaded lazily by a job (or released) is reported
    truthfully rather than from the warm-up's memory."""
    out = dict(_models)
    if out["intrinsic"] != "loading":
        try:
            out["intrinsic"] = "ready" if _intrinsic().is_loaded("careaga") else "cold"
        except Exception:  # noqa: BLE001 - module missing or model unavailable
            pass
    if out["sam2"] != "loading":
        try:
            sam = _sam_masks()
            probe = getattr(sam, "is_loaded", None) or getattr(sam.SamMasker, "is_loaded", None)
            if callable(probe):
                out["sam2"] = "ready" if probe() else "cold"
        except Exception:  # noqa: BLE001
            pass
    if out["vitmatte"] != "loading":
        try:
            probe = getattr(_matting(), "status", None)
            if callable(probe):
                out["vitmatte"] = probe()
        except Exception:  # noqa: BLE001
            pass
    for name, get in _aux_models():
        if out.get(name) != "loading":
            try:
                probe = getattr(get(), "status", None)
                if callable(probe):
                    out[name] = probe()
            except Exception:  # noqa: BLE001
                pass
    return out


def _free_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# =============================================================================
# analysis
# =============================================================================

class _Progress:
    """Adapter handed to sibling stages as `progress`. Tolerates every call shape a
    sibling might use: `(frac, msg)`, `(frac)`, `(msg)`, or keyword arguments."""

    def __init__(self, job: Job, stage: str, lo: float = 0.0, hi: float = 1.0):
        self.job, self.stage, self.lo, self.hi = job, stage, lo, hi

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        frac: Optional[float] = None
        msg: Optional[str] = None
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, str) and msg is None:
                msg = a
            elif isinstance(a, (int, float)) and not isinstance(a, bool) and frac is None:
                frac = float(a)
        p = None if frac is None else self.lo + (self.hi - self.lo) * min(1.0, max(0.0, frac))
        self.job.set_stage(self.stage, "running", progress=p, message=msg)


_OOM_RETRY_DELAY_S = 3.0


def _is_cuda_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


def _describe_error(e: BaseException) -> str:
    """A short, human message for the UI; the full traceback goes to the log. CUDA OOM
    texts are several hundred characters of allocator advice, so they are summarized."""
    text = str(e)
    if _is_cuda_oom(e):
        import re
        need = re.search(r"Tried to allocate ([\d.]+ [GM]iB)", text)
        free = re.search(r"of which ([\d.]+ [GM]iB) is free", text)
        parts = ["GPU out of memory"]
        if need and free:
            parts.append(f"(needed {need.group(1)}, {free.group(1)} free)")
        return " ".join(parts) + "; another process is using the GPU - try again in a moment or lower the detail"
    if len(text) > 300:
        text = text[:297] + "..."
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def _job_alive(job: Job) -> bool:
    return not job.deleted and registry.get(job.id) is job and os.path.isdir(job.dir)


def analyze(job: Job) -> None:
    """Run the whole analysis on the worker thread: ingest → intrinsic → segment →
    regions → groups. Every stage records progress, a human message and its seconds;
    the first exception marks the stage and the job as `error` (message preserved)
    and stops. On success the job is `ready` with all artifacts of §3.6 written. The
    job's ownership claim (`OWNER_FILE`) is released however the analysis ends."""
    try:
        _analyze(job)
    finally:
        _release_claim(job)


def _analyze(job: Job) -> None:
    job.set_status("analyzing")
    t0 = time.monotonic()
    ctx: dict[str, Any] = {}
    steps: list[tuple[str, Callable[[Job, dict[str, Any]], None]]] = [
        ("ingest", _stage_ingest),
        ("intrinsic", _stage_intrinsic),
        ("segment", _stage_segment),
        ("regions", _stage_regions),
        ("groups", _stage_groups),
    ]
    for stage, fn in steps:
        if not _job_alive(job):
            log.info("job %s deleted mid-analysis, stopping", job.id)
            job.pending_image = None
            _free_cuda()
            return
        job.set_stage(stage, "running", progress=0.0)
        try:
            try:
                with gpu_lock:
                    _touch_activity()
                    fn(job, ctx)
            except Exception as e:  # noqa: BLE001
                if not _is_cuda_oom(e) or not _job_alive(job):
                    raise
                # The card is shared: free what we can, give the neighbour a moment, retry once.
                log.warning("job %s: stage %s hit CUDA OOM, retrying once", job.id, stage)
                _free_cuda()
                job.set_stage(stage, "running", progress=0.0, message="GPU memory is tight, retrying")
                time.sleep(_OOM_RETRY_DELAY_S)
                with gpu_lock:
                    _touch_activity()
                    fn(job, ctx)
            if not _job_alive(job):
                # Deleted while the stage ran: its files are gone, do not record anything.
                log.info("job %s deleted during stage %s, stopping", job.id, stage)
                job.pending_image = None
                _free_cuda()
                return
            job.set_stage(stage, "done")
        except Exception as e:  # noqa: BLE001 - reported to the user, never crashes the worker
            job.pending_image = None   # ingest already wrote the original; nothing needs the array now
            if not _job_alive(job):
                # A stage that writes into a directory deleted under it fails with ENOENT;
                # that is not an error worth reporting (and there is nowhere to report it).
                log.info("job %s deleted during stage %s (%s), stopping", job.id, stage, type(e).__name__)
                _free_cuda()
                return
            log.exception("job %s: stage %s failed", job.id, stage)
            msg = _describe_error(e)
            job.set_stage(stage, "error", message=msg)
            for later, _ in steps[[s for s, _ in steps].index(stage) + 1:]:
                job.set_stage(later, "skipped")
            job.update(timings={**job.meta["timings"], "total_s": round(time.monotonic() - t0, 3)})
            job.set_status("error", error=f"{stage}: {msg}")
            _free_cuda()
            return
    job.pending_image = None
    job.update(timings={**job.meta["timings"], "total_s": round(time.monotonic() - t0, 3)})
    job.set_status("ready")
    _free_cuda()


def _stage_ingest(job: Job, ctx: dict[str, Any]) -> None:
    job.set_stage("ingest", "running", 0.1, "Reading the image")
    img = job.pending_image
    if img is None:
        img = imageio.load_image(job.path(job.original_file))
    h, w = img.shape[:2]
    ww, wh = imageio.fit_size(w, h, config.WORK_LONG_SIDE)
    pw, ph = imageio.fit_size(w, h, config.PREVIEW_LONG_SIDE)
    job.set_stage("ingest", "running", 0.4, f"Working copy at {ww}×{wh}")
    work = imageio.resize_to(img, (ww, wh))
    imageio.save_image(job.path("work.png"), work)
    if not os.path.exists(job.path("preview.jpg")):
        imageio.save_image(job.path("preview.jpg"), imageio.resize_to(img, (pw, ph)), quality=90)
    job.update(image={"width": w, "height": h, "work_width": ww, "work_height": wh,
                      "preview_width": pw, "preview_height": ph})
    ctx["work"] = work
    job.set_stage("ingest", "running", 1.0, f"{w}×{h} → work {ww}×{wh}, preview {pw}×{ph}")


def _stage_intrinsic(job: Job, ctx: dict[str, Any]) -> None:
    opts = job.options
    label = {"auto": "Intrinsic v2.1", "careaga": "Intrinsic v2.1", "heuristic": "heuristic"}.get(opts.intrinsic, opts.intrinsic)
    job.set_stage("intrinsic", "running", 0.02, f"Separating paint from light · {label}")
    mod = _intrinsic()
    res = mod.decompose(ctx["work"], method=opts.intrinsic, progress=_Progress(job, "intrinsic", 0.05, 0.85))
    if res.albedo.shape[:2] != ctx["work"].shape[:2]:
        raise RuntimeError(f"intrinsic returned {res.albedo.shape[:2]}, expected {ctx['work'].shape[:2]}")
    job.set_stage("intrinsic", "running", 0.9, "Saving albedo, shading and residual")
    imageio.save_f16(job.path("albedo.npy"), res.albedo)
    imageio.save_f16(job.path("shading.npy"), res.shading)
    imageio.save_f16(job.path("residual.npy"), res.residual)
    layers = mod.layers_for_display(res)
    for k in ("albedo", "shading", "residual"):
        imageio.save_image(job.path("layers", f"{k}.jpg"), layers[k], quality=92)
    job.update(intrinsic_method=res.method)
    # Segment on exactly the albedo that is stored (float16): every later group edit
    # reads it back from albedo.npy, and a regroup must see the numbers the analysis saw.
    ctx["albedo"] = res.albedo.astype(np.float16).astype(np.float32)
    ctx["residual"] = res.residual.astype(np.float16).astype(np.float32)
    ctx["shading"] = res.shading.astype(np.float16).astype(np.float32)     # the junk pruning's shadow test
    job.set_stage("intrinsic", "running", 1.0, f"Decomposed with {res.method}")


def _stage_segment(job: Job, ctx: dict[str, Any]) -> None:
    detail = job.options.detail
    job.set_stage("segment", "running", 0.02, f"Finding parts with SAM 2 · {detail}")
    masker = _sam_masks().SamMasker()
    masks = masker.generate(ctx["work"], detail=detail, progress=_Progress(job, "segment", 0.05, 0.98))
    ctx["masks"] = masks
    job.set_stage("segment", "running", 1.0, f"{len(masks)} part proposals")


def _stage_regions(job: Job, ctx: dict[str, Any]) -> None:
    job.set_stage("regions", "running", 0.02, "Merging proposals into regions")
    extras: list = []
    if job.options.detail != "fast":
        extras = _find_extras(job, ctx)
    labels, info = _hierarchy().build_regions(ctx["work"], ctx["albedo"], ctx["masks"], detail=job.options.detail,
                                              progress=_Progress(job, "regions", 0.05, 0.85), extra=extras or None)
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    if labels.shape != ctx["work"].shape[:2]:
        raise RuntimeError(f"regions returned {labels.shape}, expected {ctx['work'].shape[:2]}")
    if labels.min() < 0:
        raise RuntimeError("regions left unassigned pixels (-1)")
    n_parts = 0
    if job.options.detail != "fast":
        labels, info, n_parts = _recover_parts(job, ctx, labels, info)
    labels, info, n_cut, fg = _matte_cut(job, ctx, labels, info)
    ctx["part_masks"] = []
    if job.options.detail != "fast":
        labels, info = _detect_parts(job, ctx, labels, info, fg)
        labels, info, fg = _subject_matte(job, ctx, labels, info, fg)
    kinds = _backdrop_flags(job, ctx, labels, info, fg)
    ctx["fg"] = fg
    _write_labels(job, labels, ctx["work"])
    ctx["labels"], ctx["region_info"] = labels, info
    n_text = sum(1 for d in info if d.get("source") == "text")
    n_named = sum(1 for d in info if d.get("source") in ("wheel", "named"))
    n_small = sum(1 for d in info if d.get("source") == "small")
    n_kinds = len({d.get("part_kind") for d in info if d.get("part_kind")})
    n_inst = sum(1 for d in info if d.get("part_kind"))
    notes = []
    if n_text:
        notes.append(f"{n_text} lettering")
    if n_named + n_small:
        notes.append(f"{n_named + n_small} small and named parts")
    if n_parts:
        notes.append(f"{n_parts} small part{'s' if n_parts != 1 else ''} recovered")
    if n_kinds:
        notes.append(f"{n_inst} detected part{'s' if n_inst != 1 else ''} of {n_kinds} kind{'s' if n_kinds != 1 else ''}")
    if kinds is not None:
        notes.append(f"{int((kinds > 0).sum())} backdrop regions")
    extra = (" · " + " · ".join(notes)) if notes else ""
    job.set_stage("regions", "running", 1.0, f"{int(labels.max()) + 1} regions{extra}")


#: Wall-clock budget of the lettering / named-part prompts of one image (after Florence-2
#: itself): the regions stage holds the GPU lock, so a text-dense photo (a label sheet gives
#: 350 OCR quads) must not hold every other job and export. Normal photos need 0.3-0.8 s.
EXTRAS_BUDGET_S = 6.0


def _find_extras(job: Job, ctx: dict[str, Any]) -> list:
    """Lettering and named-part masks for `hierarchy.build_regions` (Balanced and Max):
    Florence-2's OCR quads and phrase boxes, prompted as SAM boxes (`smallparts.find_extras`,
    given `EXTRAS_BUDGET_S` of wall clock, after which the remaining prompts are skipped and
    logged). An enhancement, never a failure: without the model (or its weights), on a CUDA
    OOM or on any error the stage runs without extras, with a logged warning."""
    try:
        masker = _sam_masks().SamMasker()
        prompt = getattr(masker, "prompt_boxes", None)
        find = getattr(_smallparts(), "find_extras", None)
        if not (callable(prompt) and callable(find)):
            return []
        job.set_stage("regions", "running", 0.03, "Reading lettering and naming parts (Florence-2)")
        analysis = _florence().analyse(ctx["work"])
        if not analysis:
            return []
        job.set_stage("regions", "running", 0.04, f"Prompting SAM on {len(analysis.get('ocr', []))} words and "
                                                  f"{len(analysis.get('grounding', []))} part boxes")
        return list(find(ctx["work"], ctx["albedo"], analysis, prompt, budget_s=EXTRAS_BUDGET_S))
    except Exception as e:  # noqa: BLE001 - see docstring
        if _is_cuda_oom(e):
            log.warning("job %s: lettering and named parts skipped, the GPU is full", job.id)
        else:
            log.exception("job %s: lettering and named parts failed, building the regions without them", job.id)
        _free_cuda()
        return []


def _recover_parts(job: Job, ctx: dict[str, Any], labels: np.ndarray, info: list[dict]
                   ) -> tuple[np.ndarray, list[dict], int]:
    """`hierarchy.recover_parts` with SAM point prompts (Balanced and Max): small parts of
    their own colour that the automatic proposals missed get a region. An enhancement,
    never a failure: any error (a CUDA OOM on the shared card included) keeps the
    partition as it was, with a logged warning."""
    recover = getattr(_hierarchy(), "recover_parts", None)
    try:
        masker = _sam_masks().SamMasker()
        prompt = getattr(masker, "prompt_parts", None)
        if not (callable(recover) and callable(prompt)):
            return labels, info, 0
        job.set_stage("regions", "running", 0.92, "Looking for small parts SAM missed")
        out, out_info, n = recover(ctx["work"], ctx["albedo"], labels, info, prompter=prompt,
                                   progress=_Progress(job, "regions", 0.92, 0.98))
        out = np.ascontiguousarray(out, dtype=np.int32)
        if out.shape != labels.shape or out.min() < 0 or len(out_info) != int(out.max()) + 1:
            raise RuntimeError("part recovery returned an inconsistent partition")
        return out, out_info, int(n)
    except Exception as e:  # noqa: BLE001 - see docstring
        if _is_cuda_oom(e):
            log.warning("job %s: part recovery skipped, the GPU is full", job.id)
        else:
            log.exception("job %s: part recovery failed, keeping the partition", job.id)
        _free_cuda()
        return labels, info, 0


def _matte_cut(job: Job, ctx: dict[str, Any], labels: np.ndarray, info: list[dict]
               ) -> tuple[np.ndarray, list[dict], int, Optional[np.ndarray]]:
    """The foreground matte's first part of the regions stage: BiRefNet's matte and the
    regions cut along its contour (`hierarchy.cut_on_matte`). Returns ``(labels, info, n_cut,
    fg)``. An enhancement, never a failure: without BiRefNet (or its weights), on a CUDA OOM or
    any error the partition is kept, ``fg`` is None and the border rule decides the
    background (logged)."""
    cut = getattr(_hierarchy(), "cut_on_matte", None)
    decide = getattr(_grouping(), "backdrop_decisions", None)
    if not (callable(cut) and callable(decide)):
        return labels, info, 0, None
    try:
        job.set_stage("regions", "running", 0.96, "Telling the object from its backdrop (BiRefNet)")
        fg = _foreground().fg_prob(ctx["work"])
        if fg is None:
            return labels, info, 0, None
        out, out_info, n_cut = cut(labels, info, fg, ctx["albedo"])
        out = np.ascontiguousarray(out, dtype=np.int32)
        if out.shape != labels.shape or out.min() < 0 or len(out_info) != int(out.max()) + 1:
            raise RuntimeError("the matte cut returned an inconsistent partition")
        return out, out_info, int(n_cut), np.asarray(fg, np.float32)
    except Exception as e:  # noqa: BLE001 - see docstring
        if _is_cuda_oom(e):
            log.warning("job %s: foreground matte skipped, the GPU is full", job.id)
        else:
            log.exception("job %s: foreground matte failed, keeping the partition and the border rule", job.id)
        _free_cuda()
        return labels, info, 0, None


def _backdrop_flags(job: Job, ctx: dict[str, Any], labels: np.ndarray, info: list[dict],
                    fg: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Every region's backdrop decision (`grouping.backdrop_decisions`, stored as
    ``info[i]["bg"]`` for the groups stage; a detected part is always object), or None
    without a matte (the border rule then decides). Any error keeps the border rule (logged)."""
    decide = getattr(_grouping(), "backdrop_decisions", None)
    if fg is None or not callable(decide):
        return None
    try:
        opts = job.options
        kinds = np.asarray(decide(labels, ctx["albedo"], info, fg, delta_e=opts.delta_e, max_groups=opts.max_groups),
                           np.int8)
        for d in info:
            d["bg"] = 0 if d.get("part_kind") else int(kinds[int(d["id"])])
            kinds[int(d["id"])] = d["bg"]
        return kinds
    except Exception:  # noqa: BLE001 - an enhancement, never a failure of the analysis
        log.exception("job %s: backdrop decisions failed, the border rule decides the background", job.id)
        for d in info:
            d.pop("bg", None)
        return None


#: Wall-clock budget of the detected-part prompts of one image (after the detector itself):
#: the regions stage holds the GPU lock. Normal photos need 0.2-0.6 s.
PARTS_BUDGET_S = 6.0


def _detect_parts(job: Job, ctx: dict[str, Any], labels: np.ndarray, info: list[dict],
                  fg: Optional[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    """Detected parts (Balanced and Max, after the matte cut): Florence-2's caption picks the
    class vocabulary, OWLv2 finds the parts' boxes, SAM is prompted on them and the gated masks
    are stamped as regions of their own (`smallparts.find_kind_parts`, `stamp_parts`), which
    the groups stage keeps as one group per kind. The masks stay in ``ctx["part_masks"]`` for
    the junk pruning. An enhancement, never a failure: without the models (or their weights),
    on a CUDA OOM or on any error the partition is kept (logged)."""
    sp = _smallparts()
    find = getattr(sp, "find_kind_parts", None)
    stamp = getattr(sp, "stamp_parts", None)
    try:
        masker = _sam_masks().SamMasker()
        prompt = getattr(masker, "prompt_boxes", None)
        if not (callable(find) and callable(stamp) and callable(prompt)):
            return labels, info
        job.set_stage("regions", "running", 0.97, "Naming the parts people personalise (OWLv2)")
        cap = getattr(_florence(), "caption", None)
        caption = cap(ctx["work"]) if callable(cap) else None
        ctx["caption"] = caption
        pd = _partdetect()
        kw: dict[str, Any] = {}
        if callable(getattr(pd, "detect_in", None)):
            # the wheel second look: each wheel's brake caliper, on this partition
            kw.update(zoom=pd.detect_in, labels=labels, info=info)
        parts, rep = find(ctx["work"], ctx["albedo"], fg, caption, pd.detect, prompt, budget_s=PARTS_BUDGET_S, **kw)
        log.info("job %s parts: class %s (%r), %d detections, %d prompts, %d parts (%d calipers)", job.id,
                 rep.get("class"), rep.get("caption"), rep.get("detections", 0), rep.get("jobs", 0), len(parts),
                 rep.get("calipers", 0))
        if not parts:
            return labels, info
        lab = imageio.linear_to_lab(np.ascontiguousarray(ctx["albedo"], dtype=np.float32))
        out, out_info, srep = stamp(labels, info, parts, lab)
        out = np.ascontiguousarray(out, dtype=np.int32)
        if out.shape != labels.shape or out.min() < 0 or len(out_info) != int(out.max()) + 1:
            raise RuntimeError("the part stamping returned an inconsistent partition")
        ctx["part_masks"] = [np.asarray(p.mask, bool) for p in parts]
        return out, out_info
    except Exception as e:  # noqa: BLE001 - see docstring
        if _is_cuda_oom(e):
            log.warning("job %s: detected parts skipped, the GPU is full", job.id)
        else:
            log.exception("job %s: detected parts failed, keeping the partition", job.id)
        ctx["part_masks"] = []
        _free_cuda()
        return labels, info


def _subject_matte(job: Job, ctx: dict[str, Any], labels: np.ndarray, info: list[dict],
                   fg: Optional[np.ndarray]) -> tuple[np.ndarray, list[dict], Optional[np.ndarray]]:
    """The matte keeps one subject (Balanced and Max, after the detected parts; a car, a
    motorcycle or a bicycle by the caption): the matte pixels of another object of the photo
    that the matte took in (`subject.other_objects`: the red coupe parked behind the Torana,
    whose door panel was in the Torana's paint group) are set to backdrop and the regions are
    cut along them (`subject.cut_off`), so the backdrop decisions that follow
    give that object to the background. Detected parts and their regions are never touched.
    An enhancement, never a failure: any error keeps the partition and the matte (logged)."""
    sub, sp = _subject(), _smallparts()
    if fg is None or not callable(getattr(sub, "other_objects", None)):
        return labels, info, fg
    if sp.object_class(ctx.get("caption")) not in getattr(sub, "SINGLE_CLASSES", ()):
        return labels, info, fg
    try:
        prompt = getattr(_sam_masks().SamMasker(), "prompt_boxes", None)
        if not callable(prompt):
            return labels, info, fg
        part_ids = [int(d["id"]) for d in info if d.get("part_kind")]
        keep = list(ctx.get("part_masks") or [])
        if part_ids:
            keep.append(np.isin(labels, part_ids))
        rep: dict = {}
        others = sub.other_objects(ctx["work"], fg, prompt, keep=keep, report=rep)
        if not others.any():
            return labels, info, fg
        keep_all = np.zeros(labels.shape, bool)
        for m in keep:
            keep_all |= np.asarray(m, bool)
        others = sub.whole_regions(labels, others, keep=keep_all)
        fg2 = np.asarray(fg, np.float32).copy()
        fg2[others] = 0.0
        out, out_info, n_cut = sub.cut_off(labels, info, others, ctx["albedo"])
        out = np.ascontiguousarray(out, dtype=np.int32)
        if out.shape != labels.shape or out.min() < 0 or len(out_info) != int(out.max()) + 1:
            raise RuntimeError("the subject cut returned an inconsistent partition")
        log.info("job %s subject: %d px of other objects to the backdrop (%d regions cut), %s", job.id,
                 int(others.sum()), n_cut, rep)
        return out, out_info, fg2
    except Exception as e:  # noqa: BLE001 - see docstring
        if _is_cuda_oom(e):
            log.warning("job %s: the subject check skipped, the GPU is full", job.id)
        else:
            log.exception("job %s: the subject check failed, keeping the matte", job.id)
        _free_cuda()
        return labels, info, fg


def _stage_groups(job: Job, ctx: dict[str, Any]) -> None:
    """Cluster the regions, then refine the grouping for the engine: absorb washed-out
    paint, carve decals, snap the paint's edges (ViTMatte, or the guided-filter fallback),
    lock other materials in the paint's colour and mark own-coloured objects. Rewrites the
    label map (the snap moves boundaries) and writes the island and protect masks.

    The refinement is an enhancement: when it fails on an unusual photo (anything but a
    CUDA OOM, which the stage retries), the plain clustering is kept and the job is treated
    as unrefined (no masks), rather than the whole analysis failing. The refinement ends
    with the junk pruning (`junk.prune_junk`: shadow slivers and part rims into the group they
    are a lighting variant of; tiny backdrop crumbs into their backdrop neighbour on a
    product shot), which the seed records so that a regroup runs it again."""
    opts = job.options
    job.set_stage("groups", "running", 0.05, "Clustering regions by paint color")
    regions, groups, group_map = _grouping().group_regions(ctx["labels"], ctx["albedo"], ctx["region_info"],
                                                          max_groups=opts.max_groups, delta_e=opts.delta_e,
                                                          photo_rgb_u8=ctx["work"])
    for name in (USER_FLAGS_FILE, *[FULLRES_ALBEDO_FILE.format(method=m) for m in ("careaga", "heuristic")]):
        # a new analysis: the user's per-region flags and the full-res drift of an earlier one are void
        with contextlib.suppress(FileNotFoundError):
            os.remove(job.path(name))
    prune = _prune_params(list(groups))
    kw: dict[str, Any] = {"residual": ctx.get("residual")}
    if prune is not None:
        shading = ctx.get("shading")
        if shading is None and os.path.isfile(job.path("shading.npy")):
            shading = imageio.load_f16(job.path("shading.npy"))
        kw.update(shading=shading, fg=ctx.get("fg"), part_masks=list(ctx.get("part_masks") or []), prune=prune)
    try:
        res = _refine().refine_groups(ctx["work"], ctx["albedo"], ctx["labels"], regions, groups, group_map,
                                      _matting().snap_labels, progress=_Progress(job, "groups", 0.1, 0.95), **kw)
        labels = np.ascontiguousarray(res.labels, dtype=np.int32)
        if labels.shape != ctx["work"].shape[:2] or labels.min() < 0:
            raise RuntimeError("group refinement returned an incomplete label map")
    except Exception as e:  # noqa: BLE001 - see docstring
        if _is_cuda_oom(e):
            raise
        log.exception("job %s: group refinement failed; keeping the plain clustering (no engine masks)", job.id)
        _free_cuda()
        for name in (*[f"{m}.npy" for m in MASK_FILES], SEED_FILE):
            with contextlib.suppress(FileNotFoundError):
                os.remove(job.path(name))                   # from an earlier analysis of this job
        _write_grouping(job, list(regions), list(groups), group_map)
        note = _apply_ignore_default(job, list(groups))
        msg = f"{len(groups)} color groups from {len(regions)} regions · edge refinement skipped"
        if note["ignore_background"] is False:
            msg += f" · background kept paintable ({round(100 * note['background_share'])} % of the image)"
        job.set_stage("groups", "running", 1.0, msg)
        return
    _write_labels(job, labels, ctx["work"])
    _write_masks(job, res.islands, res.protect)
    origin = getattr(res, "origin", None)
    if origin is not None and len(origin) == int(labels.max()) + 1:
        parts = [int(d["id"]) for d in ctx["region_info"] if d.get("source") == "part"]
        n_in = int(ctx["labels"].max()) + 1
        bg = np.zeros(n_in, np.int8)
        sources = ["sam"] * n_in
        tags: dict[int, dict] = {}
        for d in ctx["region_info"]:
            if 0 <= int(d.get("id", -1)) < n_in:
                bg[int(d["id"])] = int(d.get("bg", 0) or 0)
                sources[int(d["id"])] = str(d.get("source", "sam"))
                if d.get("part_kind"):
                    tags[int(d["id"])] = {"kind": str(d["part_kind"]), "label": str(d.get("part_label", "")),
                                          "plural": str(d.get("part_plural", "")),
                                          "instance": int(d.get("part_instance", 0))}
        fg = ctx.get("fg")
        _write_seed(job, ctx["labels"], origin, parts, bg if bg.any() else None, sources, part_tags=tags,
                    fg=None if fg is None else np.asarray(fg) > 0.5, part_masks=list(ctx.get("part_masks") or []),
                    prune=prune if "prune" in kw else None)
    _write_grouping(job, list(res.regions), list(res.groups), res.group_map)
    report = dict(res.report or {})
    report.update(_apply_ignore_default(job, list(res.groups)))
    log.info("job %s groups: %s", job.id, report)
    job.set_stage("groups", "running", 1.0, groups_message(len(res.groups), len(res.regions),
                                                            int(ctx["labels"].max()) + 1, report))


#: "Ignore background" starts off when the background is fragmented like a scene: the
#: background groups are more than BG_IGNORE_MAX_GROUPS of all groups, or at least
#: BG_IGNORE_MANY of them cover more than BG_IGNORE_MAX_SHARE of the image. The matte picks
#: one salient object, so in a street scene or a showroom every other subject is flagged
#: background too, and a studio that opens with 20 of 29 rows locked is no use; the share
#: alone is not a scene (a pair of sneakers on a white backdrop is 78 % background in two
#: groups and must stay ignored). Measured: the product shots stay on (their backdrop covers
#: 0.55-0.78 of the image in 1-21 of 7-41 groups), the street scene opens off (0.83 of the
#: image, 20-22 of 29-31 groups from run to run, both rules) and so does a model kit in a
#: diorama city (0.86 in 19 of 38 groups: the second rule; its buildings, tanks and road are
#: paintable like the street's taxi, and one click ignores them).
BG_IGNORE_MAX_SHARE = 0.75
BG_IGNORE_MAX_GROUPS = 2 / 3
BG_IGNORE_MANY = 6


def default_ignore_background(groups: list[ColorGroup]) -> tuple[bool, float]:
    """The "Ignore background" default of a fresh analysis and the background's share of
    the image: on unless the background groups are more than `BG_IGNORE_MAX_GROUPS` of the
    groups, or at least `BG_IGNORE_MANY` of them cover more than `BG_IGNORE_MAX_SHARE` of
    the image (a scene with several subjects, not a product on a big plain backdrop)."""
    bg = [g for g in groups if g.is_background]
    share = float(sum(g.area_frac for g in bg))
    if not bg:
        return True, share
    scene = len(bg) > BG_IGNORE_MAX_GROUPS * len(groups) or (len(bg) >= BG_IGNORE_MANY and share > BG_IGNORE_MAX_SHARE)
    return not scene, share


def _prune_params(groups: list[ColorGroup]) -> Any:
    """The junk pruning's parameters for a fresh analysis whose plain clustering is
    ``groups`` (`junk.DEFAULT`, with the backdrop crumbs merged on a product shot, where the
    background is ignored by default; a scene's background groups are the other subjects and
    stay as they are), or None when the module is not there."""
    params = getattr(_junk(), "JunkParams", None)
    if params is None:
        return None
    scene = not default_ignore_background(groups)[0]
    return params(backdrop_crumbs=not scene)


def _apply_ignore_default(job: Job, groups: list[ColorGroup]) -> dict[str, Any]:
    """Set the job's `ignore_background` for this analysis (`default_ignore_background`) and
    return what the stage message needs."""
    ignore, share = default_ignore_background(groups)
    job.update(ignore_background=ignore)
    return {"ignore_background": ignore, "background_share": round(share, 4)}


def groups_message(n_groups: int, n_regions: int, n_input: int, report: dict[str, Any]) -> str:
    """The groups stage's summary line. The region count is the final one (the one the
    summary tile shows); when refinement changed it from the regions stage's count, the
    message says how, so the two stages' numbers add up."""
    head = f"{n_groups} color groups from {n_regions} regions"
    carved = int(report.get("decal_regions", 0) or 0) + int(report.get("part_regions", 0) or 0)
    other = n_regions - n_input - carved             # regions the edge snap emptied (negative)
    notes = []
    if carved:
        notes.append(f"{n_input} + {carved} carved out")
    if other:
        notes.append(f"{-other} merged away at the edges" if other < 0 else f"{other} added")
    if notes:
        head += f" ({', '.join(notes)})"
    parts = [head]
    named = report.get("parts") or []
    if named:
        parts.append(f"{len(named)} part{'s' if len(named) != 1 else ''} named "
                     f"({', '.join(str(p.get('name')) for p in named[:4])}{', ...' if len(named) > 4 else ''})")
    pruned = [m for m in report.get("pruned") or [] if m.get("rule") != "backdrop_crumbs"]
    if pruned:
        parts.append(f"{len(pruned)} lighting sliver{'s' if len(pruned) != 1 else ''} folded in")
    snap_name = {"vitmatte": "edges snapped with ViTMatte", "guided": "edges snapped (guided filter)"}.get(report.get("snap"))
    if snap_name:
        parts.append(snap_name)
    if report.get("decal_px"):
        parts.append("decals kept apart")
    if report.get("locked"):
        n = len(report["locked"])
        parts.append(f"{n} other material{'s' if n != 1 else ''} locked")
    if report.get("background"):
        n = len(report["background"])
        parts.append(f"{n} background group{'s' if n != 1 else ''}")
    if report.get("ignore_background") is False:
        parts.append(f"background kept paintable ({round(100 * float(report.get('background_share', 0)))} % of the image)")
    return " · ".join(parts)


def _save_npy(path: str, arr: np.ndarray) -> None:
    """np.save through a temp file and os.replace: a reader racing an edit sees the old
    file or the new one, never a torn one."""
    tmp = f"{path}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _save_png(path: str, arr: np.ndarray) -> None:
    """A PNG written through a temp file and os.replace: the studio re-reads the id maps right
    after an edit, and a reader must see the old file or the new one, never a torn one."""
    tmp = f"{path}.{os.getpid()}-{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        f.write(imageio.encode_png(np.ascontiguousarray(arr)))
    os.replace(tmp, path)


#: The display layers (``layers/<name>.png``) an edit leaves to be drawn when they are asked for
#: (`display_layer`): the studio shows them on their own tabs only, and drawing and encoding them
#: took 45-65 ms of every commit and Remove. The analysis draws them.
DISPLAY_LAYERS: dict[str, tuple[str, str]] = {"regions": ("layers", "regions.png"), "edges": ("layers", "edges.png"),
                                              "groups": ("layers", "groups.png")}


def _drop_display(job: Job, names: Iterable[str]) -> None:
    """Delete display layers an edit made stale (drawn again on demand by `display_layer`)."""
    for name in names:
        with contextlib.suppress(FileNotFoundError):
            os.remove(job.path(*DISPLAY_LAYERS[name]))


def display_layer(job: Job, name: str) -> Optional[str]:
    """The path of display layer ``name`` ('regions', 'edges' or 'groups'), drawn now when an edit
    left it to be drawn on demand: inside the job's edit lock, so no edit changes the label map or
    the groups while it is drawn and it never shows a mix of two. None when the job has nothing to
    draw it from yet; PipelineError(503) when an edit holds the job for too long."""
    p = job.path(*DISPLAY_LAYERS[name])
    if os.path.isfile(p):
        return p
    if job.status != "ready":
        return None
    with _edit_quiet(job) as got:
        if not got:
            raise PipelineError(503, "job is being edited, try again", retry_after=1)
        if os.path.isfile(p):
            return p
        try:
            if name == "groups":
                img = groups_display(np.load(job.path("group_map.npy")).astype(np.int32), job.groups())
            else:
                labels = np.load(job.path("labels.npy")).astype(np.int32)
                img = regions_display(labels) if name == "regions" else \
                    edges_display(imageio.load_image(job.path("work.png")), labels)
        except FileNotFoundError:
            return None
        _save_png(p, img)
    return p


def _write_labels(job: Job, labels: np.ndarray, work: np.ndarray, display: bool = True) -> None:
    """Persist the region label map with its id PNG and (``display``, the analysis) its display
    PNGs, each written atomically; an edit (``display`` False) deletes the display PNGs instead, to be
    drawn on demand (`display_layer`)."""
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    _save_npy(job.path("labels.npy"), labels)
    _save_png(job.path("ids", "regions.png"), encode_region_ids(labels))
    if display:
        _save_png(job.path("layers", "regions.png"), regions_display(labels))
        _save_png(job.path("layers", "edges.png"), edges_display(work, labels))
    else:
        _drop_display(job, ("regions", "edges"))


MASK_FILES = ("islands", "protect")


def _write_masks(job: Job, islands: Optional[np.ndarray], protect: Optional[np.ndarray]) -> None:
    """Persist the engine's per-pixel masks (bool, working resolution). ``None`` leaves
    the file as it is."""
    for name, mask in (("islands", islands), ("protect", protect)):
        if mask is not None:
            _save_npy(job.path(f"{name}.npy"), np.ascontiguousarray(mask, dtype=bool))


def _is_refined(job: Job) -> bool:
    """True for jobs whose groups stage wrote the engine masks (analysed with group
    refinement); older jobs keep their original behaviour under group edits."""
    return all(os.path.isfile(job.path(f"{name}.npy")) for name in MASK_FILES)


SEED_FILE = "regroup.npz"


def _write_seed(job: Job, labels_input: np.ndarray, origin: np.ndarray, parts: Optional[list[int]] = None,
                bg: Optional[np.ndarray] = None, sources: Optional[list[str]] = None,
                part_tags: Optional[dict[int, dict]] = None, fg: Optional[np.ndarray] = None,
                part_masks: Optional[list[np.ndarray]] = None, prune: Any = None) -> None:
    """Persist what `refine.regroup_refined` needs to reproduce the analysis's grouping:
    the regions stage's label map (before decals and the boundary snap), for every final
    region the input region it descends from (-1 for none), the input regions that are
    parts cut out of chromatic regions (`refine.isolate_parts`), every input region's
    backdrop decision (`grouping.backdrop_decisions`; none for the border rule), every
    input region's source (the highlight absorb's decal guard reads it) and detected-part
    tag (``part_tags``: input region id -> kind, label, plural, instance), and for the junk
    pruning the analysis ran (``prune``, its `junk.JunkParams`; none for an analysis without
    it) the object mask ``fg`` (bool) and the detected parts' SAM masks (packed bits)."""
    n_in = int(np.asarray(labels_input).max()) + 1
    tags = part_tags or {}
    kind = [str(tags.get(i, {}).get("kind", "")) for i in range(n_in)]
    arrays: dict[str, Any] = {
        "labels": np.ascontiguousarray(labels_input, dtype=np.int32),
        "origin": np.ascontiguousarray(origin, dtype=np.int32),
        "parts": np.asarray(parts or [], dtype=np.int32),
        "bg": np.asarray(bg if bg is not None else [], dtype=np.int8),
        "sources": np.asarray([str(v) for v in (sources or [])], dtype=str),
    }
    if any(kind):
        arrays.update(part_kind=np.asarray(kind, dtype=str),
                      part_label=np.asarray([str(tags.get(i, {}).get("label", "")) for i in range(n_in)], dtype=str),
                      part_plural=np.asarray([str(tags.get(i, {}).get("plural", "")) for i in range(n_in)], dtype=str),
                      part_instance=np.asarray([int(tags.get(i, {}).get("instance", -1)) for i in range(n_in)], np.int32))
    if prune is not None:
        from dataclasses import asdict
        arrays["prune"] = np.asarray(json.dumps(asdict(prune)))
        if fg is not None:
            arrays["fg"] = np.packbits(np.asarray(fg, bool).ravel())
        masks = [np.asarray(m, bool) for m in (part_masks or []) if np.asarray(m).shape == np.asarray(labels_input).shape]
        if masks:
            arrays["part_masks"] = np.stack([np.packbits(m.ravel()) for m in masks])
    tmp = job.path(f"{SEED_FILE}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, job.path(SEED_FILE))


class Seed(tuple):
    """What `_load_seed` returns: ``(labels, origin, parts, bg, sources)`` by position (the
    older shape) plus ``part_tags``, ``fg``, ``part_masks`` and ``prune`` by name."""

    def __new__(cls, labels, origin, parts, bg, sources, part_tags=None, fg=None, part_masks=None, prune=None):
        self = super().__new__(cls, (labels, origin, parts, bg, sources))
        self.labels, self.origin, self.parts, self.bg, self.sources = labels, origin, parts, bg, sources
        self.part_tags = part_tags or {}
        self.fg = fg
        self.part_masks = part_masks or []
        self.prune = prune
        return self


def _load_seed(job: Job, shape: tuple[int, ...]) -> Optional[Seed]:
    """The regroup seed of a refined job (`Seed`), or None (older jobs, unreadable file);
    ``parts`` is empty and ``bg`` / ``sources`` None for a seed written before they were
    recorded, ``part_tags`` empty and ``prune`` None for a seed from before detected parts and
    the junk pruning (a regroup then groups as that analysis did)."""
    p = job.path(SEED_FILE)
    if not os.path.isfile(p):
        return None
    h, w = (int(v) for v in tuple(shape)[:2])
    try:
        with np.load(p) as z:
            labels_input = z["labels"].astype(np.int32)
            origin = z["origin"].astype(np.int32)
            parts = [int(v) for v in z["parts"]] if "parts" in z.files else []
            bg = z["bg"].astype(np.int8) if "bg" in z.files and z["bg"].size else None
            sources = [str(v) for v in z["sources"]] if "sources" in z.files and z["sources"].size else None
            tags: dict[int, dict] = {}
            if "part_kind" in z.files:
                kinds, labs, plurals = z["part_kind"], z["part_label"], z["part_plural"]
                inst = z["part_instance"]
                for i, k in enumerate(kinds.tolist()):
                    if k:
                        tags[i] = {"kind": str(k), "label": str(labs[i]), "plural": str(plurals[i]),
                                   "instance": int(inst[i])}
            prune = None
            if "prune" in z.files:
                params = getattr(_junk(), "JunkParams", None)
                if params is not None:
                    d = json.loads(str(z["prune"]))
                    known = set(params.__dataclass_fields__)
                    # a rule added after this analysis stays off, so the regroup reruns the pruning it ran
                    for k, off in (getattr(_junk(), "LEGACY_OFF", None) or {}).items():
                        d.setdefault(k, off)
                    prune = params(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known})
            fg = np.unpackbits(z["fg"])[: h * w].reshape(h, w).astype(bool) if "fg" in z.files else None
            masks = [np.unpackbits(row)[: h * w].reshape(h, w).astype(bool) for row in z["part_masks"]] \
                if "part_masks" in z.files else []
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("job %s: ignoring unreadable %s (%s)", job.id, SEED_FILE, e)
        return None
    if labels_input.shape != tuple(shape) or labels_input.min() < 0 or origin.ndim != 1:
        log.warning("job %s: ignoring %s that does not match the label map", job.id, SEED_FILE)
        return None
    n_in = int(labels_input.max()) + 1
    if bg is not None and len(bg) != n_in:
        bg = None
    if sources is not None and len(sources) != n_in:
        sources = None
    tags = {i: t for i, t in tags.items() if i < n_in}
    return Seed(labels_input, origin, parts, bg, sources, part_tags=tags, fg=fg, part_masks=masks, prune=prune)


def _write_grouping(job: Job, regions: list[Region], groups: list[ColorGroup], group_map: np.ndarray,
                    drop_paint: Iterable[int] = (), remap: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
                    display: bool = True) -> None:
    """Persist regions/groups/group_map (+ id and display PNGs) and publish `groups`.

    The saved `mapping` is keyed by group id, and the grouping module renumbers ids
    (0..G-1 by area) on every merge/split/move/regroup, so the mapping is carried over
    by *region membership* (`remap_mapping`) rather than by id; the paint of the old groups
    in ``drop_paint`` is not carried (a merge ``into`` a group keeps that group's own paint,
    not a smaller member's). An edit that knows where every group went passes ``remap`` (the
    current mapping -> the new one) instead: a user part's carve and Remove keep each group's
    own paint. The groups get their panel view first (`grouping.annotate_groups`: which are
    minor and next to what). An edit (``display`` False) leaves the groups' display PNG to be drawn
    on demand (`display_layer`)."""
    group_map = np.ascontiguousarray(group_map, dtype=np.int32)
    if group_map.min() < 0:
        raise RuntimeError("group map has unassigned pixels (-1)")
    rule = _annotate(groups, regions, group_map)
    _save_npy(job.path("group_map.npy"), group_map)
    _save_png(job.path("ids", "groups.png"), encode_group_ids(group_map))
    if display:
        _save_png(job.path("layers", "groups.png"), groups_display(group_map, groups))
    else:
        _drop_display(job, ("groups",))
    tmp = job.path(f"regions.json.{os.getpid()}-{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in regions], f)
    os.replace(tmp, job.path("regions.json"))
    _invalidate(job.id)
    drop = {str(int(i)) for i in drop_paint}
    valid = {str(g.id) for g in groups}
    with job.lock:
        current = dict(job.meta.get("mapping") or {})
        if remap is not None:
            job.meta["mapping"] = {k: v for k, v in remap(current).items() if k in valid}
        else:
            old_groups = [ColorGroup.from_dict(g) for g in job.meta.get("groups") or []]
            job.meta["mapping"] = remap_mapping({k: v for k, v in current.items() if k not in drop},
                                                old_groups, groups, regions)
        job.meta[PANEL_RULE_KEY] = rule
    job.set_groups(groups, regions_count=len(regions))


def _annotate(groups: list[ColorGroup], regions: list[Region], group_map: np.ndarray) -> Optional[int]:
    """The Groups panel's view of the groups, in place (`grouping.annotate_groups`); a
    grouping module without it (a test fake) leaves the groups as they are. Returns the panel
    rule the view now follows (`grouping.PANEL_RULE`), or None when nothing was annotated."""
    annotate = getattr(_grouping(), "annotate_groups", None)
    if not callable(annotate):
        return None
    try:
        annotate(groups, regions, np.asarray(group_map))
    except Exception:  # noqa: BLE001 - a view hint, never a failure of an edit
        log.exception("annotating the groups failed; the panel shows them unsectioned")
        return None
    return getattr(_grouping(), "PANEL_RULE", None)


#: job.json key: the panel rule (`grouping.PANEL_RULE`) the stored groups' view was made with.
PANEL_RULE_KEY = "panel_rule"


def refresh_panel_view(job: Job) -> bool:
    """A ready job whose groups carry the view of an older panel rule (`grouping.PANEL_RULE`;
    none recorded: before the rule was versioned) is annotated again, in memory: a job analysed
    when size alone made a group minor hid real small parts under the collapsed Minor divider
    (the Ducati's 690 px gold preload adjuster) until some edit re-annotated it. `job.json` is
    not rewritten (opening a job must not change it); the next edit or state save stores the new
    view with the rule. Skipped while an edit of the job runs (it annotates anyway) and on any
    error (logged). Returns True when the view was refreshed."""
    rule = getattr(_grouping(), "PANEL_RULE", None)
    annotate = getattr(_grouping(), "annotate_groups", None)
    if rule is None or not callable(annotate):
        return False
    with job.lock:
        raw = job.meta.get("groups")
        if job.meta.get("status") != "ready" or not raw or job.meta.get(PANEL_RULE_KEY) == rule:
            return False
    if not job.edit_lock.acquire(blocking=False):
        return False
    try:
        groups = [ColorGroup.from_dict(g) for g in raw]
        group_map = np.load(job.path("group_map.npy"))
        annotate(groups, _load_regions(job), group_map)
        with job.lock:
            if job.meta.get("groups") is not raw:
                return False                                  # written meanwhile (and annotated)
            job.meta["groups"] = [g.to_dict() for g in groups]
            job.meta[PANEL_RULE_KEY] = rule
        return True
    except Exception:  # noqa: BLE001 - a view hint, never a failure of opening a job
        log.exception("job %s: refreshing the groups' panel view failed", job.id)
        return False
    finally:
        job.edit_lock.release()


def remap_mapping(mapping: dict[str, Any], old_groups: list[ColorGroup], new_groups: list[ColorGroup],
                  regions: list[Region]) -> dict[str, Any]:
    """Carry a `{gid: hex|null}` mapping across a regrouping by region membership.

    For every old group with a mapping entry, the color moves to the new group that
    holds the (area-weighted) majority of the old group's regions; ties and groups
    whose regions scattered (no new group holds more than half) are dropped. When
    two old groups land on the same new group the larger (first-listed) old group
    wins. Region ids are stable across edits (contract §3.2) so this is exact.
    """
    if not mapping:
        return {}
    area = {r.id: max(int(r.area), 1) for r in regions}
    region_to_new = {rid: g.id for g in new_groups for rid in g.region_ids}
    out: dict[str, Any] = {}
    for old in old_groups:
        key = str(old.id)
        if key not in mapping:
            continue
        weights: Counter[int] = Counter()
        total = 0
        for rid in old.region_ids:
            w = area.get(rid, 1)
            total += w
            gid = region_to_new.get(rid)
            if gid is not None:
                weights[gid] += w
        if not weights or total <= 0:
            continue
        best, w = weights.most_common(1)[0]
        if w * 2 <= total or str(best) in out:
            continue
        out[str(best)] = mapping[key]
    return out


# =============================================================================
# id / display encoders
# =============================================================================

def encode_region_ids(labels: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 with `id = R + 256·G + 65536·B` (exact for ids below 2^24)."""
    ids = labels.astype(np.int64)
    out = np.empty(labels.shape + (3,), np.uint8)
    out[..., 0] = ids & 255
    out[..., 1] = (ids >> 8) & 255
    out[..., 2] = (ids >> 16) & 255
    return out


def decode_region_ids(rgb: np.ndarray) -> np.ndarray:
    """Inverse of `encode_region_ids` -> int32 HxW."""
    r = rgb.astype(np.int32)
    return r[..., 0] + 256 * r[..., 1] + 65536 * r[..., 2]


def encode_group_ids(group_map: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 with R = group id (clamped to 0..255), G = B = 0."""
    out = np.zeros(group_map.shape + (3,), np.uint8)
    out[..., 0] = np.clip(group_map, 0, 255).astype(np.uint8)
    return out


def regions_display(labels: np.ndarray) -> np.ndarray:
    """Each region in a random but deterministic saturated color (uint8 RGB)."""
    n = int(labels.max()) + 1
    rng = np.random.default_rng(1234)
    hsv = np.stack([rng.random(n) * 179, 120 + rng.random(n) * 135, 150 + rng.random(n) * 105], 1).astype(np.uint8)
    import cv2
    lut = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]
    return lut[np.clip(labels, 0, n - 1)]


def groups_display(group_map: np.ndarray, groups: list[ColorGroup]) -> np.ndarray:
    """Each pixel painted with its group's albedo color (uint8 RGB)."""
    n = max(int(group_map.max()) + 1, len(groups), 1)
    lut = np.full((n, 3), 128, np.uint8)
    for g in groups:
        if 0 <= g.id < n:
            lut[g.id] = imageio.to_uint8(imageio.hex_to_rgb01(g.albedo_hex))
    return lut[np.clip(group_map, 0, n - 1)]


def edges_display(image_rgb_u8: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Region boundaries drawn in cyan over a slightly dimmed copy of the image."""
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    out = (image_rgb_u8.astype(np.float32) * 0.8).astype(np.uint8)
    out[edge] = (40, 230, 255)
    return out


# =============================================================================
# cached layers and renderers
# =============================================================================

_layers_cache: "OrderedDict[str, dict[str, np.ndarray]]" = OrderedDict()
_renderer_cache: "OrderedDict[str, Any]" = OrderedDict()
# The edit generation each cached renderer was built at (``get_renderer``): a full-resolution export reuses
# the preview's renderer for its working-resolution reference only when it holds the export's own snapshot.
_renderer_gen: dict[str, int] = {}
_cache_lock = threading.Lock()
_LAYERS_CACHE_SIZE = 4
_RENDERER_CACHE_SIZE = 3
# Per-job edit generation (a sequence lock): odd while a group edit is writing the job's
# grouping files, bumped again when it is done. A reader that loads the layers and the
# groups (a renderer build, a full-resolution export) checks that the generation did not
# move while it read, so it never mixes a group map from before an edit with the groups
# from after it, and never caches such a mix (a renderer built during a merge was kept
# with the old group list).
_cache_gen: dict[str, int] = {}
_STABLE_WAIT_S = 2.0


def _require_ready(job: Job) -> None:
    if job.status != "ready":
        raise PipelineError(409, f"job is {job.status}, not ready")


def _generation(job_id: str) -> int:
    with _cache_lock:
        return _cache_gen.get(job_id, 0)


@contextlib.contextmanager
def _writing(job: Job, drop: bool = True):
    """The section of a group edit that writes the job's grouping (files, meta): the edit
    generation is odd inside it and moves on when it ends, and the cached layers and
    renderer are dropped on both sides (``drop``; a lock toggle updates them in place)."""
    with _cache_lock:
        _cache_gen[job.id] = _cache_gen.get(job.id, 0) + 1
        if drop:
            _layers_cache.pop(job.id, None)
            _renderer_cache.pop(job.id, None)
    try:
        yield
    finally:
        with _cache_lock:
            _cache_gen[job.id] = _cache_gen.get(job.id, 0) + 1
            if drop:
                _layers_cache.pop(job.id, None)
                _renderer_cache.pop(job.id, None)


@contextlib.contextmanager
def _edit_quiet(job: Job, timeout: float = _STABLE_WAIT_S * 5):
    """The job's edit lock, taken for a read when edits keep racing it, but never waited on
    for longer than ``timeout`` (a reader must not deadlock with an edit that waits for the
    GPU lock); yields whether it was taken."""
    got = job.edit_lock.acquire(timeout=timeout)
    try:
        yield got
    finally:
        if got:
            job.edit_lock.release()


def _stable_generation(job: Job) -> int:
    """The job's edit generation once no edit is writing (waits up to _STABLE_WAIT_S, then
    for the edit itself)."""
    deadline = time.monotonic() + _STABLE_WAIT_S
    while True:
        gen = _generation(job.id)
        if gen % 2 == 0:
            return gen
        if time.monotonic() > deadline:
            with _edit_quiet(job):
                return _generation(job.id)
        time.sleep(0.005)


def _snapshot(job: Job, build: Callable[[dict[str, Optional[np.ndarray]], list[ColorGroup]], Any]
              ) -> tuple[Any, int]:
    """``build(layers, groups)`` on the layers and groups of one edit generation, rebuilt
    when an edit wrote the grouping meanwhile. Returns ``(result, generation)``. When edits
    keep racing it, the last attempt runs inside the job's edit lock, where no edit writes.
    Callers must not hold ``gpu_lock`` (an edit may hold the edit lock while it waits for it)."""
    for _ in range(4):
        gen = _stable_generation(job)
        out = build(dict(load_layers(job)), effective_groups(job))
        if _generation(job.id) == gen:
            return out, gen
    with _edit_quiet(job):
        gen = _generation(job.id)
        return build(dict(load_layers(job)), effective_groups(job)), gen


def ignores_background(job: Job) -> bool:
    """The job's "Ignore background" setting (on unless the record says otherwise)."""
    return bool(job.meta.get("ignore_background", True))


def effective_groups(job: Job) -> list[ColorGroup]:
    """The groups as the renderer, the exports and the mapping suggestions see them: copies of
    the stored groups with every background group locked while the job ignores its
    background. The stored flags (the user's own locks) are never changed by this."""
    groups = job.groups()
    if ignores_background(job):
        for g in groups:
            if g.is_background:
                g.locked = True
    return groups


def save_state(job: Job, fields: dict[str, Any]) -> None:
    """Persist the validated fields of `PUT /state` (`validate_state`). A change of
    `ignore_background` changes the groups' effective locks, so it is applied like a group
    edit: under the job's edit lock and inside an edit generation, and a cached renderer takes
    the new flags in place. A renderer being built meanwhile read the old flags before the
    write and is not cached (the generation moved), so no preview keeps painting, or
    ignoring, the background after the switch."""
    if "ignore_background" not in fields:
        job.update(**fields)
        return
    with job.edit_lock:
        with _writing(job, drop=False):
            job.update(**fields)
            _update_renderer_flags(job, effective_groups(job), None)


def load_layers(job: Job) -> dict[str, Optional[np.ndarray]]:
    """Working-resolution arrays `albedo`, `shading`, `residual` (float32 linear),
    `labels`, `group_map` (int32) and the engine masks `islands`, `protect` (bool, or None
    for a job analysed before they existed). Cached per job (small LRU); invalidated by
    group edits, and never cached from files a concurrent edit was rewriting. Raises
    PipelineError(409) before analysis is complete."""
    _require_ready(job)
    with _cache_lock:
        if job.id in _layers_cache:
            _layers_cache.move_to_end(job.id)
            return _layers_cache[job.id]
        gen = _cache_gen.get(job.id, 0)
    for attempt in range(5):
        try:
            layers = {
                "albedo": imageio.load_f16(job.path("albedo.npy")),
                "shading": imageio.load_f16(job.path("shading.npy")),
                "residual": imageio.load_f16(job.path("residual.npy")),
                "labels": np.load(job.path("labels.npy")).astype(np.int32),
                "group_map": np.load(job.path("group_map.npy")).astype(np.int32),
            }
            break
        except FileNotFoundError as e:
            raise PipelineError(404, f"missing artifact: {os.path.basename(str(e.filename))}") from e
        except (OSError, ValueError, EOFError) as e:
            # Edits replace these files atomically, but a reader can still land between
            # the write of one and the next; the pair is consistent again within a few ms.
            if attempt == 4:
                raise PipelineError(503, "job is being edited, try again") from e
            time.sleep(0.05)
    for name in MASK_FILES:
        layers[name] = _load_mask(job, name, layers["group_map"].shape)
    with _cache_lock:
        if gen % 2 == 0 and _cache_gen.get(job.id, 0) == gen:
            _layers_cache[job.id] = layers
            while len(_layers_cache) > _LAYERS_CACHE_SIZE:
                _layers_cache.popitem(last=False)
    return layers


def _load_mask(job: Job, name: str, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    """A bool mask artifact, or None when the job has none (or an unusable one)."""
    p = job.path(f"{name}.npy")
    if not os.path.isfile(p):
        return None
    try:
        m = np.load(p).astype(bool)
    except (OSError, ValueError, EOFError) as e:
        log.warning("job %s: ignoring unreadable %s.npy (%s)", job.id, name, e)
        return None
    if m.shape != tuple(shape):
        log.warning("job %s: ignoring %s.npy of shape %s (group map %s)", job.id, name, m.shape, tuple(shape))
        return None
    return m


def get_renderer(job: Job):
    """The engine `Renderer` holding this job's layers on the GPU. Cached per job and
    dropped on regroup/merge/split/move (or when more than a few jobs are active). Built
    from the layers and groups of one edit generation (see `_snapshot`), and cached only
    when no edit wrote the grouping while it was built."""
    _require_ready(job)
    with _cache_lock:
        r = _renderer_cache.get(job.id)
        if r is not None:
            _renderer_cache.move_to_end(job.id)
            return r

    def build(layers: dict[str, Optional[np.ndarray]], groups: list[ColorGroup]):
        with _render_lock:
            return _engine().Renderer(layers["albedo"], layers["shading"], layers["residual"],
                                      layers["group_map"], groups,
                                      islands=layers.get("islands"), protect=layers.get("protect"))

    renderer, gen = _snapshot(job, build)
    with _cache_lock:
        if _cache_gen.get(job.id, 0) == gen:
            cached = _renderer_cache.get(job.id)
            if cached is not None:
                return cached                 # another request built the same state meanwhile
            _renderer_cache[job.id] = renderer
            _renderer_gen[job.id] = gen
            while len(_renderer_cache) > _RENDERER_CACHE_SIZE:
                _renderer_cache.popitem(last=False)
    return renderer


def _update_renderer_flags(job: Job, groups: list[ColorGroup], protect: Optional[np.ndarray]) -> None:
    """After a lock / background / rename edit: hand the cached renderer the new flags
    (and protect mask) in place, so the next preview does not pay for a new renderer;
    drop it when it cannot take them."""
    with _cache_lock:
        renderer = _renderer_cache.get(job.id)
    if renderer is None:
        return
    update = getattr(renderer, "update_groups", None)
    try:
        if not callable(update):
            raise TypeError("renderer has no update_groups")
        with _render_lock:
            update(groups, protect=protect)
    except Exception as e:  # noqa: BLE001 - a rebuild on the next render is always correct
        log.info("job %s: renderer rebuilt after a group edit (%s)", job.id, e)
        with _cache_lock:
            _renderer_cache.pop(job.id, None)


def _invalidate(job_id: str) -> None:
    with _cache_lock:
        # a build that started before this call must not be cached (parity kept: an edit
        # writing its files keeps the generation odd until it is done)
        _cache_gen[job_id] = _cache_gen.get(job_id, 0) + 2
        _layers_cache.pop(job_id, None)
        _renderer_cache.pop(job_id, None)


def _install_layers(job: Job, layers: dict[str, Optional[np.ndarray]], labels: np.ndarray, group_map: np.ndarray,
                    protect: Optional[np.ndarray]) -> None:
    """After an edit wrote its files (still inside the job's edit lock, the generation even again):
    the job's cached layers are the arrays it wrote, with the layers an edit never changes (albedo,
    shading, residual, islands) as they were, so the next commit or render does not load the job's
    files again (20-30 ms of every commit). ``protect`` None keeps the protect mask as it was.
    Nothing is installed when a reader cached the layers meanwhile or the job is not ready."""
    if job.status != "ready":
        return
    fresh = dict(layers)
    fresh["labels"] = np.ascontiguousarray(labels, np.int32)
    fresh["group_map"] = np.ascontiguousarray(group_map, np.int32)
    if protect is not None:
        fresh["protect"] = np.ascontiguousarray(protect, dtype=bool)
    with _cache_lock:
        if _cache_gen.get(job.id, 0) % 2 == 0 and job.id not in _layers_cache:
            _layers_cache[job.id] = fresh
            while len(_layers_cache) > _LAYERS_CACHE_SIZE:
                _layers_cache.popitem(last=False)


def invalidate(job: Job) -> None:
    """Drop cached arrays, the GPU renderer and the SAM 2 prompt session (the work image's
    embedding, Select part) of this job (after delete)."""
    _invalidate(job.id)
    _forget_prompts(job.id)
    _free_cuda()


def _forget_prompts(job_id: str) -> None:
    """Drop the job's SAM 2 prompt session, if SAM 2 was ever imported (never imported just for this)."""
    import sys
    mod = sys.modules.get("recolor.segmentation.sam_masks")
    forget = getattr(mod, "forget_prompts", None) if mod is not None else None
    if callable(forget):
        try:
            forget(job_id)
        except Exception:  # noqa: BLE001 - a cache, never a failure of the delete
            log.exception("job %s: dropping the prompt session failed", job_id)


# =============================================================================
# render / export
# =============================================================================

def _validate_mapping(job: Job, mapping: Any) -> Mapping:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise PipelineError(400, "mapping must be an object of group id -> hex color or null")
    valid = {g["id"] for g in job.meta.get("groups") or []}
    out: Mapping = {}
    for k, v in mapping.items():
        try:
            gid = int(k)
        except (TypeError, ValueError):
            raise PipelineError(400, f"mapping key {k!r} is not a group id") from None
        if valid and gid not in valid:
            raise PipelineError(400, f"mapping refers to unknown group {gid}")
        if v in (None, "", False):
            out[gid] = None
            continue
        if not isinstance(v, str):
            raise PipelineError(400, f"mapping value for group {gid} must be a hex color or null")
        try:
            rgb = imageio.hex_to_rgb01(v)
        except ValueError:
            raise PipelineError(400, f"mapping value {v!r} for group {gid} is not a hex color") from None
        out[gid] = imageio.rgb01_to_hex(rgb)
    return out


def _validate_options(options: Any) -> RenderOptions:
    if options is None:
        return RenderOptions()
    if not isinstance(options, dict):
        raise PipelineError(400, "options must be an object")
    o = RenderOptions.from_dict(options)
    if o.mode not in ("shift", "flat"):
        raise PipelineError(400, f"options.mode must be 'shift' or 'flat', not {o.mode!r}")
    for name, lo, hi in (("texture", 0.0, 1.0), ("feather_px", 0.0, 64.0), ("residual_tint", 0.0, 1.0),
                         ("shading_strength", 0.0, 4.0), ("saturation", 0.0, 4.0)):
        try:
            val = float(getattr(o, name))
        except (TypeError, ValueError, OverflowError):
            raise PipelineError(400, f"options.{name} must be a number") from None
        if not (lo <= val <= hi):
            raise PipelineError(400, f"options.{name} must be between {lo} and {hi}")
        setattr(o, name, val)
    o.keep_residual = bool(o.keep_residual)
    o.sharpen_edges = bool(o.sharpen_edges)
    return o


def render_preview(job: Job, mapping: Any, options: Any) -> bytes:
    """JPEG bytes of the recolored image at `config.PREVIEW_LONG_SIDE`. Mapping and
    options are validated (PipelineError 400 on bad values); the job must be ready."""
    m = _validate_mapping(job, mapping)
    o = _validate_options(options)
    renderer = get_renderer(job)
    with _render_lock:
        out = renderer.render_at(config.PREVIEW_LONG_SIDE, m, o)
    return imageio.encode_jpeg(np.ascontiguousarray(out), quality=90)


def _export_name(job: Job, quality: str, fmt: str) -> str:
    stem = os.path.splitext(job.meta["name"])[0] or "image"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60]
    return f"{stem}_recolor_{quality}_{time.strftime('%Y%m%d-%H%M%S')}.{fmt}"


def export(job: Job, mapping: Any, options: Any, quality: str, fmt: str) -> dict[str, Any]:
    """Render at working (`quality="work"`) or original (`"full"`) resolution and write
    `exports/<name>.<fmt>`. Returns `{file, width, height, ms}`.

    Full resolution re-runs the intrinsic decomposition on the original when it has at
    most `config.FULLRES_INTRINSIC_MAX_PIXELS` pixels (so the identity holds exactly)
    **and** that run uses the same method the preview was tuned with (`intrinsic_method`
    in the job meta); if the model falls back to the heuristic (CUDA OOM on a shared
    card) or the free VRAM is clearly too small to try, the working-res layers are
    guided-upsampled instead, with the residual recomputed so that
    `albedo·shading + residual == original` still holds. The result's `intrinsic` key
    says which path ran: `'careaga'` / `'heuristic'` (full-res decomposition) or
    `'upsampled'`. The group map is nearest-upsampled and snapped to the original's
    edges with the guided filter (skipped only when that would need more GPU memory
    than is sensible on a shared card).
    """
    if quality not in EXPORT_QUALITIES:
        raise PipelineError(400, f"quality must be one of {list(EXPORT_QUALITIES)}")
    if fmt not in EXPORT_FORMATS:
        raise PipelineError(400, f"format must be one of {list(EXPORT_FORMATS)}")
    m = _validate_mapping(job, mapping)
    o = _validate_options(options)
    t0 = time.perf_counter()
    try:
        if quality == "work":
            def work_render() -> tuple[np.ndarray, str]:
                renderer = get_renderer(job)
                with _render_lock:
                    return renderer.render(m, o), "work"
            out, intrinsic = _gpu_busy_retry(job, work_render)
        else:
            # the layers and groups of one edit generation, read before the GPU lock is taken
            (layers, groups), gen = _snapshot(job, lambda layers, groups: (layers, groups))

            def full_render() -> tuple[np.ndarray, str]:
                with gpu_lock:
                    _touch_activity()
                    return _render_full(job, m, o, layers, groups, gen)
            out, intrinsic = _gpu_busy_retry(job, full_render)
    finally:
        _free_cuda()
    name = _export_name(job, quality, fmt)
    os.makedirs(job.path("exports"), exist_ok=True)
    imageio.save_image(job.path("exports", name), np.ascontiguousarray(out), quality=95)
    ms = int((time.perf_counter() - t0) * 1000)
    return {"file": name, "width": int(out.shape[1]), "height": int(out.shape[0]), "ms": ms, "intrinsic": intrinsic}


EXPORT_RETRY_AFTER_S = 10


def _gpu_busy_retry(job: Job, fn: Callable[[], Any]) -> Any:
    """``fn()``; on a CUDA OOM (the card is shared) free the cache, wait a moment and try
    once more, then give up with PipelineError(503) and a retry hint instead of a 500."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        if not _is_cuda_oom(e):
            raise
        log.warning("job %s: export hit CUDA OOM, retrying once", job.id)
    _free_cuda()
    time.sleep(_OOM_RETRY_DELAY_S)
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        if not _is_cuda_oom(e):
            raise
        log.warning("job %s: export hit CUDA OOM twice; reporting the GPU as busy", job.id)
        _free_cuda()
        raise PipelineError(503, "The GPU is busy with other work right now. Try the export again in a moment, "
                                 "or export at working resolution.", retry_after=EXPORT_RETRY_AFTER_S) from e


# The VRAM a full-resolution Careaga pass needs, fitted to the peak *allocations* measured on
# the 5090 with the model resident (5.9 GB at 2.2 MP, 13.5 GB at 6.3 MP, 20.2 GB at 9.8 MP:
# 1.75 GB + 1.9 GB per MP; the earlier 1.0 + 1.5 estimate put the yellow BMW's 9.8 MP at
# 15.7 GB and sent it into an OOM with 0.3 GB to spare). The caching allocator *reserves*
# more (8.1 / 19.2 / 28.9 GB on the same passes; 7.3 / 17.1 / 25.7 GB with the allocator's
# expandable segments, which serve.py enables): where the free VRAM is below that reserved
# peak the allocator frees its cached blocks and retries, which PyTorch logs as a
# CUDACachingAllocator warning, and the pass completes. That warning is expected for a
# >= 9 MP pass whenever the card is shared (another server or the owner's app holding a few
# GB), not a failure; the fit is on the allocation. The card is shared with care: a pass
# may take at most _FULLRES_COMFORT of the free VRAM straight away (the rest stays free for
# the previews of other users); above that the segmentation models are released first and
# the plain fit decides (`_fullres_room`). A pass that fits but still OOMs falls back inside
# the intrinsic module and is then caught by the method-consistency check in `_render_full`.
_FULLRES_GB_PER_MP = 1.9
_FULLRES_GB_FLOOR = 1.75
_FULLRES_COMFORT = 0.8


def _fullres_need_gb(n_pixels: int) -> float:
    """The fitted VRAM need of a full-res model pass, in GB (the peak allocation)."""
    return _FULLRES_GB_FLOOR + _FULLRES_GB_PER_MP * n_pixels / 1e6


def _free_vram_gb() -> Optional[float]:
    """The device's free VRAM in GB, or None without CUDA (nothing to check then)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        return None
    return free / 2**30


def _fullres_intrinsic_fits(n_pixels: int) -> bool:
    """False when the free VRAM is clearly too small for a full-res model pass."""
    free = _free_vram_gb()
    if free is None:
        return True
    need = _fullres_need_gb(n_pixels)
    fits = free >= need
    if not fits:
        log.info("full-res intrinsic does not fit: %.1f GB free, ~%.1f GB needed for %.1f MP",
                 free, need, n_pixels / 1e6)
    return fits


def _fullres_fits_comfortably(n_pixels: int) -> bool:
    """False when the pass would take more than `_FULLRES_COMFORT` of the free VRAM."""
    free = _free_vram_gb()
    if free is None:
        return True
    need = _fullres_need_gb(n_pixels)
    fits = _FULLRES_COMFORT * free >= need
    if not fits:
        log.info("full-res intrinsic does not fit comfortably: %.1f GB free, ~%.1f GB needed for %.1f MP",
                 free, need, n_pixels / 1e6)
    return fits


def _fullres_room(n_pixels: int) -> bool:
    """True when the full-res model pass may run: it fits in `_FULLRES_COMFORT` of the free
    VRAM, or, when it does not, in the free VRAM after the segmentation models (SAM 2,
    ViTMatte, Florence-2, BiRefNet: about 5 GB, none of them needed by an export, all
    reloaded lazily by the next analysis) are released. A 10 MP pass on a warm server
    (five models resident, another app on the card) had 0.3 GB to spare and went through an
    OOM before its fallback; now it takes the freed card. The free VRAM is read here, right
    before the pass and device-wide (`torch.cuda.mem_get_info` counts every process), so
    other users of the card are accounted for; the caching allocator's free-and-retry
    warning can still print while the pass reserves more than is free (see the constants
    above). Called with `gpu_lock` held."""
    _free_cuda()
    if _fullres_fits_comfortably(n_pixels):
        return True
    released = _release_segmentation_models("full-res export")
    if not released:
        fits = _fullres_intrinsic_fits(n_pixels)
        if not fits:
            log.info("full-res intrinsic skipped; upsampling working-res layers")
        return fits
    _free_cuda()
    fits = _fullres_intrinsic_fits(n_pixels)
    log.info("full-res export released %s to make room; the pass %s", " + ".join(released),
             "fits now" if fits else "still does not fit, upsampling working-res layers")
    return fits


# A full-resolution decomposition is used for an export only while it agrees with the
# working-resolution albedo the preview was tuned on: at most FULLRES_MAX_DRIFT of the
# repainted pixels may differ by more than FULLRES_DRIFT_DE (CIEDE2000, compared at working
# resolution). Measured: Ducati 3.5 %, Exia 4.3 %, yellow BMW 18-19 %.
FULLRES_DRIFT_DE = 10.0
FULLRES_MAX_DRIFT = 0.10


def _albedo_at_work(layers: dict[str, Optional[np.ndarray]], albedo_full: np.ndarray) -> np.ndarray:
    """A full-resolution albedo area-downsampled to the working resolution (float32)."""
    import cv2
    h, w = layers["albedo"].shape[:2]
    return cv2.resize(np.ascontiguousarray(albedo_full, dtype=np.float32), (w, h), interpolation=cv2.INTER_AREA)


def _fullres_drift(layers: dict[str, Optional[np.ndarray]], albedo_down: np.ndarray, m: Mapping,
                   groups: list[ColorGroup]) -> float:
    """Share of the repainted pixels (working resolution) whose full-res albedo, already
    area-downsampled to the working resolution (`_albedo_at_work`), is more than
    FULLRES_DRIFT_DE from the stored working-res albedo; 0 for an empty mapping (the
    identity holds on either path)."""
    locked = {g.id for g in groups if g.locked}
    ids = [gid for gid, v in (m or {}).items() if v and gid not in locked]
    if not ids:
        return 0.0
    work = layers["albedo"]
    sel = np.isin(layers["group_map"], ids)
    if not sel.any() or albedo_down.shape[:2] != work.shape[:2]:
        return 0.0
    a = imageio.linear_to_lab(np.ascontiguousarray(work[sel], dtype=np.float32)[None])[0]
    b = imageio.linear_to_lab(np.clip(albedo_down[sel], 0.0, 1.0)[None])[0]
    return float((imageio.delta_e(a, b) > FULLRES_DRIFT_DE).mean())


# The full-resolution albedo a first full export computed, area-downsampled to the working
# resolution (float16), per intrinsic method: the drift of any later mapping is known
# without the model pass. The yellow BMW drifts 18 % for its paint, so every full export
# ran a 20 GB, 4-11 s Careaga pass and threw it away (and hit CUDA OOM on the shared card).
FULLRES_ALBEDO_FILE = "fullres_albedo_{method}.npy"


def _cached_fullres_albedo(job: Job, method: Optional[str], shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if not method:
        return None
    p = job.path(FULLRES_ALBEDO_FILE.format(method=method))
    if not os.path.isfile(p):
        return None
    try:
        down = imageio.load_f16(p)
    except (OSError, ValueError) as e:
        log.warning("job %s: ignoring unreadable %s (%s)", job.id, os.path.basename(p), e)
        return None
    return down if down.shape[:2] == tuple(shape[:2]) else None


def _store_fullres_albedo(job: Job, method: str, down: np.ndarray) -> None:
    try:
        imageio.save_f16(job.path(FULLRES_ALBEDO_FILE.format(method=method)), down)
    except OSError as e:
        log.warning("job %s: could not cache the full-res albedo (%s)", job.id, e)


def _upsample_group_map(layers: dict[str, Optional[np.ndarray]], guide: np.ndarray
                        ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """(group_map, islands, protect) at the guide's resolution. The masks travel with the
    group map as one combined label (group * 4 + island + 2 * protect), nearest-upsampled
    and snapped to the original's edges together, so a decal or a protected part keeps
    exactly its group's refined boundary. Refinement is skipped when it would need more GPU
    memory than is sensible on a shared card."""
    H, W = guide.shape[:2]
    gm = layers["group_map"]
    isl, prot = layers.get("islands"), layers.get("protect")
    code = gm.astype(np.int64) * 4
    if isl is not None:
        code += isl.astype(np.int64)
    if prot is not None:
        code += 2 * prot.astype(np.int64)
    uniq, inv = np.unique(code, return_inverse=True)
    up = filters.upsample_labels(inv.reshape(gm.shape).astype(np.int32), (W, H))
    if H * W * min(len(uniq), 32) <= _REFINE_MAX_ELEMENTS:
        try:
            up = filters.refine_labels_with_guide(up, guide, radius=4)
        except RuntimeError as e:  # CUDA OOM: keep the nearest upsample
            log.warning("label refinement skipped (%s)", e)
            _free_cuda()
    else:
        log.info("label refinement skipped: %dx%d with %d labels exceeds the memory budget", W, H, len(uniq))
    code_up = uniq[up]
    group_map = (code_up // 4).astype(np.int32)
    islands = (code_up & 1).astype(bool) if isl is not None else None
    protect = (code_up & 2).astype(bool) if prot is not None else None
    return group_map, islands, protect


def _render_full(job: Job, m: Mapping, o: RenderOptions, layers: Optional[dict[str, Optional[np.ndarray]]] = None,
                 groups: Optional[list[ColorGroup]] = None, gen: Optional[int] = None) -> tuple[np.ndarray, str]:
    """The full-resolution render and which intrinsic path produced its layers
    (`'careaga'` / `'heuristic'` for a full-res decomposition, `'upsampled'`). ``layers`` and
    ``groups`` are one consistent snapshot (`_snapshot`) of edit generation ``gen``; read here when
    not given."""
    if layers is None or groups is None:
        (layers, groups), gen = _snapshot(job, lambda layers, groups: (layers, groups))
    original = imageio.load_image(job.path(job.original_file))
    H, W = original.shape[:2]
    guide = imageio.to_float(original)
    group_map, islands, protect = _upsample_group_map(layers, guide)
    glints, neutral = _working_reference(job, layers, groups, gen, m, o)

    lin = imageio.srgb_to_linear(guide)
    albedo = shading = residual = None
    path = "upsampled"
    expected = job.meta.get("intrinsic_method") or None
    known = _cached_fullres_albedo(job, expected, layers["albedo"].shape)
    drift_known = _fullres_drift(layers, known, m, groups) if known is not None else None
    if drift_known is not None and drift_known > FULLRES_MAX_DRIFT:
        # An earlier export measured this: the full-res pass would be thrown away.
        log.info("full-res intrinsic skipped: an earlier pass drifted from the preview's albedo (%.0f %% of the "
                 "repainted pixels by more than dE %.0f); upsampling working-res layers", 100 * drift_known,
                 FULLRES_DRIFT_DE)
    elif H * W <= config.FULLRES_INTRINSIC_MAX_PIXELS and (expected != "careaga" or _fullres_room(H * W)):
        try:
            res = _intrinsic().decompose(original, method=expected or "auto")
            if res.albedo.shape[:2] != (H, W):
                log.warning("full-res intrinsic returned %s for %dx%d; upsampling working-res layers",
                            res.albedo.shape[:2], H, W)
            elif expected is not None and res.method != expected:
                # The model fell back (typically CUDA OOM on the shared card). Rendering
                # with different layers than the preview was tuned on would change the
                # look, so use the preview's own layers upsampled instead.
                log.warning("full-res intrinsic ran %r but the preview used %r; upsampling working-res layers",
                            res.method, expected)
                _free_cuda()
            else:
                down = _albedo_at_work(layers, res.albedo)
                if known is None:
                    _store_fullres_albedo(job, str(res.method), down)
                drift = _fullres_drift(layers, down, m, groups)
                if drift > FULLRES_MAX_DRIFT:
                    # The model split paint and light differently at full resolution (the
                    # yellow BMW: 19 % of the paint moved by more than dE 10 and the tail
                    # read flat with a grey patch): the export would not look like the
                    # preview, so use the preview's own layers upsampled.
                    log.info("full-res intrinsic drifted from the preview's albedo (%.0f %% of the repainted "
                             "pixels by more than dE %.0f); upsampling working-res layers", 100 * drift, FULLRES_DRIFT_DE)
                else:
                    albedo, shading, residual = res.albedo, res.shading, res.residual
                    path = str(res.method)
                del res
                _free_cuda()
        except Exception as e:  # noqa: BLE001 - fall back to upsampling, never fail the export
            log.warning("full-res intrinsic failed (%s); upsampling working-res layers", e)
            _free_cuda()
    if albedo is None:
        path = "upsampled"
        albedo = np.clip(filters.guided_filter(guide, layers["albedo"], radius=8, eps=1e-3), 0.0, 1.0)
        shading = np.clip(filters.guided_filter(guide, layers["shading"], radius=8, eps=1e-3), 0.0, None)
        residual = (lin - albedo * shading).astype(np.float32)
    # The engine's pixel distances (reflection falloff, edge bands) were tuned at the
    # working resolution: scale them with the export so it looks like the preview.
    work_long = int(max(layers["group_map"].shape))
    out = _engine().render_once(albedo.astype(np.float32), shading.astype(np.float32),
                                residual.astype(np.float32), group_map, groups, m, o,
                                islands=islands, protect=protect, reference_long_side=work_long,
                                glints=glints, neutral=neutral)
    return out, path


def _working_reference(job: Job, layers: dict[str, Optional[np.ndarray]], groups: list[ColorGroup],
                       gen: Optional[int], m: Mapping, o: RenderOptions
                       ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(glints, neutral)`` for a full-resolution export: the white paint's own glints (engine rule 7e)
    and every group's neutral-source weights (``Renderer.neutral_weights``) as the preview's renderer
    finds them on the working-resolution layers. Measured again on the full-resolution layers, which the
    decomposition splits differently, they were not the preview's (the Alpine's shoulder glint went,
    twenty sill reflections came on the upsampled layers; a group's glint share, which decides whether
    a glossy grey is white paint, moves with the residual). The preview's cached renderer is used when
    it holds this very snapshot (edit generation ``gen``), else one is built and freed again before the
    full-resolution pass. ``glints`` is None when the mapping repaints no neutral source (nothing to keep;
    the export then has none either, since it takes these weights). ``(None, None)`` when the card has no
    room: the export then measures its own."""
    engine = _engine()
    with _cache_lock:
        cached = _renderer_cache.get(job.id)
        if cached is not None and (gen is None or _renderer_gen.get(job.id) != gen):
            cached = None
    try:
        if cached is not None:
            with _render_lock:
                neutral = cached.neutral_weights()
                glints = cached.white_glints() if cached.neutral_sources(m, o) else None
            return glints, neutral
        r = engine.Renderer(layers["albedo"], layers["shading"], layers["residual"], layers["group_map"], groups,
                            islands=layers.get("islands"), protect=layers.get("protect"))
        try:
            neutral = r.neutral_weights()
            glints = r.white_glints() if r.neutral_sources(m, o) else None
            return glints, neutral
        finally:
            r.free()
    except RuntimeError as e:  # CUDA OOM on the shared card, or a renderer freed meanwhile
        log.warning("working-resolution reference skipped (%s)", e)
        _free_cuda()
        return None, None


# =============================================================================
# group edits
# =============================================================================

def _load_regions(job: Job) -> list[Region]:
    try:
        with open(job.path("regions.json"), "r", encoding="utf-8") as f:
            return [Region.from_dict(d) for d in json.load(f)]
    except FileNotFoundError as e:
        raise PipelineError(404, "missing artifact: regions.json") from e


def _int_list(value: Any, what: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PipelineError(400, f"{what} must be a non-empty list of integers")
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError):
        raise PipelineError(400, f"{what} must be a non-empty list of integers") from None


def _int_value(value: Any, what: str) -> int:
    if isinstance(value, bool) or value is None:
        raise PipelineError(400, f"{what} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise PipelineError(400, f"{what} must be an integer") from None


def _bounded_int(value: Any, what: str, lo: int, hi: int) -> int:
    """``value`` as an integer in ``lo..hi``: an int, a whole finite float (8.0) or a string of
    digits (a form field). A bool, a fraction (1.7, which ``int`` truncated), NaN, a number too
    large for anything (10**400 was stored in job.json and reused) or anything else is a 400."""
    bad = PipelineError(400, f"{what} must be an integer between {lo} and {hi}")
    if isinstance(value, bool) or value is None:
        raise bad
    if isinstance(value, float):
        if not np.isfinite(value) or not value.is_integer():
            raise bad
        value = int(value)
    elif isinstance(value, str):
        s = value.strip()
        digits = s[1:] if s[:1] in ("+", "-") else s
        if not (1 <= len(digits) <= 12 and digits.isascii() and digits.isdigit()):
            raise bad
        value = int(s)
    elif not isinstance(value, int):
        raise bad
    if not (lo <= value <= hi):
        raise bad
    return int(value)


def _bounded_float(value: Any, what: str, lo: float, hi: float) -> float:
    """``value`` as a finite number in ``lo..hi`` (strings of a form field accepted); a bool, NaN,
    an integer too large for a float (``float(10**400)`` raised OverflowError: a 500) or anything
    else is a 400."""
    bad = PipelineError(400, f"{what} must be a number between {lo:g} and {hi:g}")
    if isinstance(value, bool) or value is None:
        raise bad
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        raise bad from None
    if not (np.isfinite(v) and lo <= v <= hi):
        raise bad
    return v


# The user's own lock / background choices, per region (`user_flags.json`): a regroup builds
# new groups (with the analysis's automatic flags) and then gives every group more than
# half of whose area the user had flagged the user's choice again, the way the mapping is
# carried by region membership. A lock set by hand on the BMW's brake reservoir bracket
# came back unlocked after a regroup while its navy mapping was carried over.
USER_FLAGS_FILE = "user_flags.json"
USER_FLAG_KINDS = ("locked", "is_background")
#: ``user_flags.json`` key of the user parts (Select part, Find part): ``{kind: {"label",
#: "regions": [ids]}}``, the regroup's safety net (`userparts.apply_registry`).
USER_PARTS_KEY = "parts"


def _read_user_file(job: Job) -> dict[str, Any]:
    """The raw ``user_flags.json`` record ({} for a job without one, or an unreadable one, logged)."""
    try:
        with open(job.path(USER_FLAGS_FILE), "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("job %s: ignoring unreadable %s (%s)", job.id, USER_FLAGS_FILE, e)
        return {}
    return data if isinstance(data, dict) else {}


def _write_user_file(job: Job, data: dict[str, Any]) -> None:
    """``user_flags.json`` written atomically (temp file + rename)."""
    tmp = job.path(f"{USER_FLAGS_FILE}.{os.getpid()}-{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, job.path(USER_FLAGS_FILE))


def _load_user_flags(job: Job) -> dict[str, dict[str, bool]]:
    """``{"locked": {region_id: bool}, "is_background": {...}}``; empty for a job without any
    (or with an unreadable file, logged)."""
    out: dict[str, dict[str, bool]] = {k: {} for k in USER_FLAG_KINDS}
    data = _read_user_file(job)
    for k in USER_FLAG_KINDS:
        votes = data.get(k)
        if isinstance(votes, dict):
            for rid, v in votes.items():
                try:
                    out[k][str(int(rid))] = bool(v)
                except (TypeError, ValueError):
                    continue
    return out


def _record_user_flags(job: Job, group: ColorGroup, payload: dict[str, Any]) -> None:
    """Remember the user's lock / background choice for every region of ``group`` (the rest of
    the record, the user parts, is kept)."""
    data = _read_user_file(job)
    flags = _load_user_flags(job)
    changed = False
    for k in USER_FLAG_KINDS:
        if payload.get(k) is not None:
            for rid in group.region_ids:
                flags[k][str(int(rid))] = bool(payload[k])
            changed = True
    if not changed:
        return
    data.update(flags)
    _write_user_file(job, data)


#: The keys of a user part's record that an edit keeps as they are (`_sync_user_parts`).
USER_PART_HISTORY = ("donors", "was", "groups", "homes", "paints", "new_id")


def _load_user_parts(job: Job) -> dict[str, dict]:
    """The user parts the job records (``{kind: {"label", "regions": [ids]}}``, `USER_PARTS_KEY`),
    with, when recorded, ``donors`` (region -> pixels it gave), ``was`` (the tag of each region it
    took in whole), ``groups`` (every group it took in whole: ``{"regions", "name", "locked",
    "is_background", "part", "paint"?}``), ``homes`` (a region of a group it took only in part ->
    a region of that group left outside), ``new_id`` (the region its carve made, whose pixels'
    donors `_save_carve` keeps), ``paints`` (older records: kind -> the paint of a part group it took
    in whole) and, for a user part another one took in whole (no regions of its own then),
    ``inside`` (that one's kind). A malformed record is left out."""
    raw = _read_user_file(job).get(USER_PARTS_KEY)
    out: dict[str, dict] = {}
    if not isinstance(raw, dict):
        return out
    for kind, rec in raw.items():
        if not isinstance(rec, dict):
            continue
        try:
            item: dict[str, Any] = {"label": str(rec.get("label") or kind),
                                    "regions": [int(v) for v in rec.get("regions") or []]}
            donors = rec.get("donors")
            if isinstance(donors, dict) and donors:
                item["donors"] = {str(int(k)): int(v) for k, v in donors.items()}
            was = rec.get("was")
            if isinstance(was, dict) and was:
                item["was"] = {str(int(k)): (v if isinstance(v, dict) else None) for k, v in was.items()}
            took = rec.get("groups")
            if isinstance(took, list) and took:
                items = []
                for gr in took:
                    if not isinstance(gr, dict) or not gr.get("regions"):
                        continue
                    one = {"regions": [int(v) for v in gr["regions"]], "name": str(gr.get("name") or ""),
                           "locked": gr.get("locked") is True, "is_background": gr.get("is_background") is True,
                           "part": str(gr.get("part") or "")}
                    if isinstance(gr.get("paint"), str) and gr["paint"]:
                        one["paint"] = gr["paint"]
                    items.append(one)
                if items:
                    item["groups"] = items
            homes = rec.get("homes")
            if isinstance(homes, dict) and homes:
                item["homes"] = {str(int(k)): int(v) for k, v in homes.items()}
            if isinstance(rec.get("new_id"), int) and not isinstance(rec.get("new_id"), bool) and rec["new_id"] >= 0:
                item["new_id"] = int(rec["new_id"])
            paints = rec.get("paints")
            if isinstance(paints, dict) and paints:
                item["paints"] = {str(k): str(v) for k, v in paints.items() if isinstance(v, str) and v}
            if isinstance(rec.get("inside"), str) and rec["inside"]:
                item["inside"] = rec["inside"]
            out[str(kind)] = item
        except (TypeError, ValueError):
            continue
    return out


def _save_user_parts(job: Job, parts: dict[str, dict]) -> None:
    """Record the user parts (the lock and background choices are kept)."""
    data = _read_user_file(job)
    if not parts and USER_PARTS_KEY not in data:
        return
    data[USER_PARTS_KEY] = parts
    _write_user_file(job, data)


def carry_names(old_groups: list[ColorGroup], new_groups: list[ColorGroup], regions: list[Region]) -> int:
    """In place, after a regroup: a group the user named ("Solo seat") gives its name to the new
    group that is mostly its regions (more than half of the old group's area went there, and it
    makes more than half of the new group's), the way the mapping is carried by region
    membership; automatic names (colour names, part names) are recomputed as before. Returns
    how many names were carried."""
    grouping = _grouping()
    keep = getattr(grouping, "_keep_name", None)
    if not callable(keep):
        return 0
    area = {r.id: max(int(r.area), 1) for r in regions}
    new_of = {rid: g for g in new_groups for rid in g.region_ids}
    new_area = {g.id: sum(area.get(rid, 1) for rid in g.region_ids) for g in new_groups}
    taken: set[int] = set()
    n = 0
    for old in sorted(old_groups, key=lambda g: -g.area):
        name = keep(old)
        if not name or (old.part and grouping._is_auto_part_name(old, name)):
            continue
        weights: Counter[int] = Counter()
        total = 0
        for rid in old.region_ids:
            w = area.get(rid, 1)
            total += w
            g = new_of.get(rid)
            if g is not None:
                weights[g.id] += w
        if not weights or total <= 0:
            continue
        gid, w = weights.most_common(1)[0]
        if gid in taken or 2 * w <= total or 2 * w <= new_area.get(gid, 0):
            continue
        target = next(g for g in new_groups if g.id == gid)
        if target.name != name:
            target.name = name
            n += 1
        taken.add(gid)
    return n


def apply_user_flags(groups: list[ColorGroup], regions: list[Region], flags: dict[str, dict[str, bool]]) -> bool:
    """In place: a group more than half of whose area (by region) carries a user choice for a
    flag (``flags`` as stored in USER_FLAGS_FILE) takes the area-weighted majority of those
    choices; other groups keep the flags they have. Returns True when a flag changed."""
    area = {r.id: max(int(r.area), 1) for r in regions}
    changed = False
    for g in groups:
        total = sum(area.get(rid, 1) for rid in g.region_ids)
        for k in USER_FLAG_KINDS:
            votes = flags.get(k) or {}
            if not votes:
                continue
            yes = sum(area.get(rid, 1) for rid in g.region_ids if votes.get(str(rid)) is True)
            no = sum(area.get(rid, 1) for rid in g.region_ids if votes.get(str(rid)) is False)
            if 2 * (yes + no) > total and yes != no:          # a tie keeps the automatic flag
                value = yes > no
                if bool(getattr(g, k)) != value:
                    setattr(g, k, value)
                    changed = True
    return changed


def apply_group_edit(job: Job, kind: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Apply a user edit to the grouping and persist every affected artifact.

    kinds: `merge {group_ids}`, `split {group_id, k, mode?}` (`mode` 'colour', the default:
    k-means on the albedo; 'instances': a part group with several instances into one group
    per instance, `grouping.split_instances`), `move {region_ids, group_id}`,
    `regroup {max_groups?, delta_e?}`, `update {group_id, name?, locked?, is_background?}`
    (a name of 1..NAME_MAX_LEN characters and true / false flags; a null leaves the field).
    Bad ids or values raise PipelineError(400); the job must be ready (409 otherwise). Cached
    layers and the renderer are invalidated and a `groups` event is published.

    Edits of one job are serialised (`Job.edit_lock`, held for the whole edit): each reads
    the grouping another left, so none is lost (two lock toggles 20 ms apart used to keep
    only the second, and a toggle racing a merge wrote the pre-merge groups back). Readers
    never see a half-written edit (see `_snapshot`).

    A pixel-level split rewrites the label map. For a job analysed with group refinement
    the engine masks stay consistent: the islands are pixel facts and never change, the
    protect mask is recomputed for the new paint (and for a lock or background change),
    and a regroup re-applies the absorb rule and the material lock, as the analysis did.
    A regroup also keeps the user's own lock and background choices (`apply_user_flags`).
    """
    if kind not in GROUP_EDIT_KINDS:
        raise PipelineError(400, f"unknown edit {kind!r}")
    with job.edit_lock:
        _require_ready(job)
        return _apply_group_edit(job, kind, payload or {})


def _apply_group_edit(job: Job, kind: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    groups = job.groups()
    gids = {g.id for g in groups}

    if kind == "update":
        gid = _int_value(payload.get("group_id"), "group_id")
        target = next((g for g in groups if g.id == gid), None)
        if target is None:
            raise PipelineError(400, f"unknown group {gid}")
        # validated before anything is changed: a string name of 1..NAME_MAX_LEN characters,
        # true / false flags (as `validate_state` demands of ignore_background; "false" or "x"
        # must not become True); a null leaves that field as it is
        if "name" in payload and payload["name"] is not None:
            if not isinstance(payload["name"], str):
                raise PipelineError(400, "name must be a string")
            if not payload["name"].strip():
                raise PipelineError(400, "name must not be empty")
            if len(payload["name"].strip()) > NAME_MAX_LEN:
                raise PipelineError(400, f"name must be at most {NAME_MAX_LEN} characters")
        for key in USER_FLAG_KINDS:
            if key in payload and payload[key] is not None and not isinstance(payload[key], bool):
                raise PipelineError(400, f"{key} must be true or false")
        if "name" in payload and payload["name"] is not None:
            target.name = payload["name"].strip()
        if "locked" in payload and payload["locked"] is not None:
            target.locked = payload["locked"]
        if "is_background" in payload and payload["is_background"] is not None:
            target.is_background = payload["is_background"]
        paint_changed = any(payload.get(k) is not None for k in USER_FLAG_KINDS)
        protect = None
        if paint_changed and _is_refined(job):
            layers = load_layers(job)
            # CPU only (no gpu_lock): a lock toggle must not wait behind an analysis stage.
            *_, protect = _refine().refine_after_edit(
                imageio.load_image(job.path("work.png")), layers["albedo"], layers["labels"], _load_regions(job),
                groups, layers["group_map"], layers.get("islands"), regrouped=False)
        if paint_changed:
            _record_user_flags(job, target, payload)
        if payload.get("is_background") is not None or job.meta.get(PANEL_RULE_KEY) != getattr(_grouping(), "PANEL_RULE", None):
            # the object area and the Minor rows' parents follow the background flags (and a view
            # of an older panel rule is brought up to date with the edit that stores it)
            rule = _annotate(groups, _load_regions(job), load_layers(job)["group_map"])
            with job.lock:
                job.meta[PANEL_RULE_KEY] = rule
        # The renderer and the layers stay cached (updated in place): a lock toggle then costs
        # a normal render, not a new renderer.
        with _writing(job, drop=False):
            if protect is not None:
                _write_masks(job, None, protect)
                with _cache_lock:
                    cached = _layers_cache.get(job.id)
                    if cached is not None:
                        cached["protect"] = np.ascontiguousarray(protect, dtype=bool)
            job.set_groups(groups)
            _update_renderer_flags(job, effective_groups(job), protect)
        return

    layers = load_layers(job)
    regions = _load_regions(job)
    albedo, group_map = layers["albedo"], layers["group_map"]
    old_groups = list(groups)
    # split_group cuts regions in the label map in place: work on a copy, so the cached
    # layers (which a concurrent render may be reading) stay as they are until replaced.
    labels = layers["labels"].copy() if kind == "split" else layers["labels"]
    n_regions_before = int(labels.max()) + 1
    grouping = _grouping()
    seed = None
    protect = None
    inherit: Optional[tuple[set, Optional[str]]] = None
    dissolved = False
    drop_paint: set[int] = set()
    remap_paint: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
    labels_changed = False                     # Remove part gave a carve's pixels back (same region count)
    origin_fix: dict[int, int] = {}
    answer: Optional[dict[str, Any]] = None    # what Remove part adds to the answer (`removed_part`)

    if kind == "merge":
        ids = _int_list(payload.get("group_ids"), "group_ids")
        unknown = [i for i in ids if i not in gids]
        if unknown:
            raise PipelineError(400, f"unknown groups {unknown}")
        dissolve = payload.get("dissolve", False)
        if not isinstance(dissolve, bool):
            raise PipelineError(400, "dissolve must be true or false")
        removing = len(set(ids)) == 1 and dissolve and payload.get("into") is None
        if removing:
            # Remove part: one user part goes back where its pixels came from (`_remove_user_part`);
            # its paint goes with it (carried by membership it painted the whole group it went back to)
            part = next(g for g in groups if g.id == ids[0])
            if not _userparts().is_user_kind(part.part):
                raise PipelineError(400, "only a part made with Select part can be removed on its own")
            # the carve is undone on a copy of the label map (the cached layers stay as they are)
            labels = labels.copy()
            regions, groups, group_map, remap_paint, undo = _remove_user_part(
                job, list(regions), list(groups), np.asarray(group_map), labels, part, albedo=albedo)
            labels_changed = bool(undo["labels"])
            origin_fix = dict(undo["origin"])
            answer = {"removed_part": {"name": part.name, "restored": list(undo.get("restored") or []),
                                       "home": undo.get("home")}}
            dissolved = True
        elif len(set(ids)) < 2:
            raise PipelineError(400, "merge needs at least two distinct groups")
        merge_kw: dict[str, Any] = {}
        if payload.get("into") is not None and not removing:
            # the group the others merge into keeps its name and its own paint (or none: a painted
            # part merged into an unpainted colour painted the whole colour), and the user parts
            # merged into it join it whatever the sizes (with ``dissolve`` they are removed and their
            # pixels join it)
            into = _int_value(payload.get("into"), "into")
            if into not in ids:
                raise PipelineError(400, "into must be one of group_ids")
            before = [(r.part_kind, r.part_instance) for r in regions]
            others = {i for i in ids if i != into}
            regions = _join_target_tags(regions, groups, into, lambda r: r.group_id in others, dissolve=dissolve)
            dissolved = before != [(r.part_kind, r.part_instance) for r in regions]
            merge_kw["into"] = into
            drop_paint = set(others)
        if not removing:
            with gpu_lock:
                regions, groups, group_map = grouping.merge_groups(groups, regions, group_map, labels, ids, **merge_kw)
    elif kind == "split":
        gid = _int_value(payload.get("group_id"), "group_id")
        mode = payload.get("mode") or "colour"
        if mode not in SPLIT_MODES:
            raise PipelineError(400, f"mode must be one of {list(SPLIT_MODES)}")
        if gid not in gids:
            raise PipelineError(400, f"unknown group {gid}")
        if mode == "instances":
            target = next(g for g in groups if g.id == gid)
            if not target.part or target.part_instances < 2:
                raise PipelineError(400, "only a part group with several instances splits by instance")
            # every instance keeps the part's paint (a colour split drops a colour it cannot place)
            inherit = (set(target.region_ids), (job.meta.get("mapping") or {}).get(str(gid)))
            with gpu_lock:
                regions, groups, group_map = grouping.split_instances(groups, regions, group_map, labels, gid)
        else:
            k = _int_value(payload.get("k", 2), "k")
            if not (2 <= k <= 8):
                raise PipelineError(400, "k must be between 2 and 8")
            with gpu_lock:
                regions, groups, group_map = grouping.split_group(groups, regions, group_map, labels, albedo, gid, k)
    elif kind == "move":
        rids = _int_list(payload.get("region_ids"), "region_ids")
        gid = _int_value(payload.get("group_id"), "group_id")
        known = {r.id for r in regions}
        unknown = [i for i in rids if i not in known]
        if unknown:
            raise PipelineError(400, f"unknown regions {unknown[:8]}")
        if gid not in gids:
            raise PipelineError(400, f"unknown group {gid}")
        before = [(r.part_kind, r.part_instance) for r in regions]
        moving = set(rids)
        regions = _join_target_tags(regions, groups, gid, lambda r: r.id in moving)
        dissolved = before != [(r.part_kind, r.part_instance) for r in regions]
        with gpu_lock:
            regions, groups, group_map = grouping.move_regions(groups, regions, group_map, labels, rids, gid)
    else:  # regroup
        opts = job.options
        # the same bounds as a new job's options (`parse_analysis_options`); a stored option out of
        # them (an older server kept 10**400) falls back to the default instead of failing
        if "max_groups" in payload and payload["max_groups"] is not None:
            max_groups = _bounded_int(payload["max_groups"], "max_groups", 1, MAX_GROUPS_LIMIT)
        elif "max_groups" in payload:
            max_groups = None
        else:
            try:
                max_groups = None if opts.max_groups is None else \
                    _bounded_int(opts.max_groups, "max_groups", 1, MAX_GROUPS_LIMIT)
            except PipelineError:
                max_groups = None
        if payload.get("delta_e") is not None:
            delta_e = _bounded_float(payload["delta_e"], "delta_e", DELTA_E_MIN, DELTA_E_MAX)
        else:
            try:
                delta_e = _bounded_float(opts.delta_e, "delta_e", DELTA_E_MIN, DELTA_E_MAX)
            except PipelineError:
                delta_e = AnalysisOptions().delta_e
        seed = _load_seed(job, labels.shape) if _is_refined(job) else None
        with gpu_lock:
            if seed is not None:
                # Reproduce the analysis: cluster the regions stage's (pre-snap) regions
                # again and carry the result over (refine.regroup_refined), with the detected
                # parts' groups and the junk pruning when the analysis had them.
                kw: dict[str, Any] = {"parts": seed[2]} if seed[2] else {}
                if seed[3] is not None:
                    kw["bg"] = seed[3]
                if seed[4] is not None:
                    kw["sources"] = seed[4]
                if seed.part_tags:
                    kw["part_tags"] = seed.part_tags
                if seed.prune is not None:
                    # a region the user locked or (un)flagged background joins only a host that ends
                    # with that choice (dissolved into another group, its choice would lose its
                    # majority there; a pruned sliver of the group the user flagged goes back into it)
                    kw.update(prune=seed.prune, shading=layers.get("shading"), fg=seed.fg,
                              part_masks=seed.part_masks, user_flags=_load_user_flags(job))
                regions, groups, group_map, protect = _refine().regroup_refined(
                    imageio.load_image(job.path("work.png")), albedo, seed[0], seed[1], labels, regions,
                    layers.get("islands"), max_groups=max_groups, delta_e=delta_e,
                    residual=layers.get("residual"), **kw)
            else:
                regions, groups, group_map = grouping.regroup(regions, labels, albedo, max_groups, delta_e,
                                                              photo_rgb_u8=imageio.load_image(job.path("work.png")))
        job.update(options={**opts.to_dict(), "max_groups": max_groups, "delta_e": delta_e})

    user_parts_moved = restored = False
    if kind in ("merge", "split", "move"):
        # a user part's regions follow the group they are in now (merged into its neighbour it is
        # gone, merged into another part it is an instance of it)
        regions, groups, group_map, user_parts_moved = _follow_user_parts(list(regions), list(groups),
                                                                          np.asarray(group_map), labels)

    if _is_refined(job) and not (kind == "regroup" and seed is not None):
        regrouped = kind == "regroup"
        # Only a regroup's absorb rule touches the GPU; the protect mask is CPU numpy.
        with (gpu_lock if regrouped else contextlib.nullcontext()):
            regions, groups, group_map, protect = _refine().refine_after_edit(
                imageio.load_image(job.path("work.png")), albedo, labels, list(regions), list(groups),
                np.asarray(group_map), layers.get("islands"), regrouped=regrouped)
    if kind == "regroup":
        regions, groups = list(regions), list(groups)
        registry = _load_user_parts(job)
        if registry:
            # the safety net: every user part the job records is one group of its own again
            regions, groups, group_map, restored = _userparts().apply_registry(regions, groups, labels, registry)
            regions, groups = list(regions), list(groups)
        carry_names(old_groups, groups, regions)
        if (apply_user_flags(groups, regions, _load_user_flags(job)) or restored) and _is_refined(job):
            # a lock decides what the paint is: the protect mask follows the user's flags
            *_, protect = _refine().refine_after_edit(
                imageio.load_image(job.path("work.png")), albedo, labels, regions, groups,
                np.asarray(group_map), layers.get("islands"), regrouped=False)
    with _writing(job):
        if _is_refined(job) and protect is not None:
            _write_masks(job, None, protect)
        if int(labels.max()) + 1 != n_regions_before or labels_changed:
            # split_group cut regions at the pixel level, or Remove part gave a carve's pixels back:
            # persist the new label map
            _write_labels(job, labels, imageio.load_image(job.path("work.png")), display=False)
        _write_grouping(job, list(regions), list(groups), np.asarray(group_map), drop_paint=drop_paint,
                        remap=remap_paint, display=False)
        if user_parts_moved or restored or dissolved or (kind in ("merge", "split", "move") and _load_user_parts(job)):
            # the registry and the regroup seed follow the user parts' regions and tags
            _sync_user_parts(job, list(regions), origin_fix=origin_fix)
        if inherit is not None and inherit[1]:
            with job.lock:
                mapping = dict(job.meta.get("mapping") or {})
                for g in groups:
                    if g.region_ids and set(g.region_ids) <= inherit[0]:
                        mapping[str(g.id)] = inherit[1]
                job.meta["mapping"] = mapping
            job.save()
    _install_layers(job, layers, labels, np.asarray(group_map), protect)
    return answer


# =============================================================================
# interactive segmentation (Select part, Find part) and user parts
# =============================================================================

#: How long a prompt waits for the GPU while an analysis stage or a full-resolution export holds
#: it (a Balanced SAM stage holds it about 5 s) before it answers 503 with `PROMPT_RETRY_AFTER_S`.
PROMPT_GPU_WAIT_S = 1.5
PROMPT_RETRY_AFTER_S = 2
#: Find part: the phrase's length and how many candidates come back.
FIND_TEXT_MAX = 60
FIND_MAX = 5
#: Two candidates whose masks overlap more than this are one.
FIND_DEDUP_IOU = 0.8
#: At most this many part groups the phrase names come first among the candidates (``existing``).
FIND_EXISTING_MAX = 3
#: At most this many detector phrases: the text, then its synonyms and the vocabulary's prompts.
FIND_MAX_PHRASES = 4
#: Detector phrases OWLv2 answers better than the bare word (on the Ducati "mirror" found nothing,
#: "rear view mirror" the mirror first, 0.20); the vocabulary's prompts of the part kind the text
#: names are added too (`_find_phrases`).
FIND_SYNONYMS: dict[str, tuple[str, ...]] = {
    "mirror": ("rear view mirror", "side mirror"),
    "caliper": ("brake caliper",),
    "spring": ("coil spring",),
    "disc": ("brake disc",),
    "rim": ("wheel rim",),
    "tyre": ("tire",),
    "exhaust": ("exhaust pipe", "muffler"),
    "headlight": ("headlamp",),
    "light": ("headlight", "tail light"),
    "grill": ("front grille",),
    "grille": ("front grille",),
    "seat": ("saddle",),
    "logo": ("emblem", "badge"),
    "badge": ("emblem", "logo"),
    "emblem": ("badge", "logo"),
    "lace": ("shoelaces",),
}
#: Spellings folded together when a phrase is matched against the part groups' names.
FIND_SPELLINGS = {"calliper": "caliper", "tire": "tyre", "grill": "grille", "disk": "disc", "tyres": "tyre"}


def _interactive():
    return _mod("recolor.segmentation.interactive")


def _userparts():
    return _mod("recolor.segmentation.userparts")


#: While a prompt waits for the GPU it checks this often whether its client went away.
PROMPT_CANCEL_POLL_S = 0.1


def _cancelled(check: Callable[[], bool]) -> bool:
    """``check()``, False when it fails (a client check is advisory, never a failure of the prompt)."""
    try:
        return bool(check())
    except Exception:  # noqa: BLE001
        return False


class PromptCancelled(PipelineError):
    """The client of a prompt went away while it waited for the GPU (nobody reads the answer)."""

    def __init__(self) -> None:
        super().__init__(409, "the request was cancelled")


@contextlib.contextmanager
def _prompt_gpu(what: str = "Select part", cancelled: Optional[Callable[[], bool]] = None):
    """The GPU lock for an interactive prompt: waited for at most `PROMPT_GPU_WAIT_S` (an analysis
    stage or a full-resolution export holds it), else PipelineError(503) with a retry hint. With
    ``cancelled`` (the client went away: the studio aborts a superseded click) the wait is given up
    at the next `PROMPT_CANCEL_POLL_S` with `PromptCancelled`, and no model runs for it, so a burst
    of clicks during an analysis does not hold a server thread each for the whole wait. The prompt
    counts as activity for the idle unload, before and after it runs."""
    deadline = time.monotonic() + PROMPT_GPU_WAIT_S
    waited = False
    while True:
        left = deadline - time.monotonic()
        step = left if cancelled is None else min(PROMPT_CANCEL_POLL_S, left)
        if gpu_lock.acquire(timeout=max(0.0, step)):
            break
        waited = True
        if cancelled is not None and _cancelled(cancelled):
            raise PromptCancelled()
        if time.monotonic() >= deadline:
            raise PipelineError(503, f"The GPU is busy with an analysis or an export; {what} works again in a moment",
                                retry_after=PROMPT_RETRY_AFTER_S)
    try:
        if waited and cancelled is not None and _cancelled(cancelled):
            raise PromptCancelled()               # it went away while this waited: nothing to run
        _touch_activity()
        yield
    finally:
        _touch_activity()
        gpu_lock.release()


def _work_size(job: Job) -> tuple[int, int]:
    img = job.meta.get("image") or {}
    w, h = int(img.get("work_width") or 0), int(img.get("work_height") or 0)
    if w <= 0 or h <= 0:
        raise PipelineError(409, "the job has no working image yet")
    return w, h


def _prompt_session(job: Job):
    """``(session, computed)``: the job's SAM 2 prompt session (`SamMasker.prompt_session`: the
    work image's embedding, cached for the next prompts). Raises PipelineError(404) without a
    work image or for a job deleted meanwhile (whose embedding is dropped again: the delete's
    `invalidate` may have run before this prompt cached it) and PipelineError(503) when SAM 2
    cannot run here."""
    p = job.path("work.png")
    try:
        st = os.stat(p)
    except FileNotFoundError as e:
        raise PipelineError(404, "missing artifact: work.png") from e
    sam = _sam_masks()
    cls = getattr(sam, "SamMasker", None)
    masker = cls.instance() if callable(getattr(cls, "instance", None)) else (cls() if cls is not None else None)
    if masker is None or not callable(getattr(masker, "prompt_session", None)):
        raise PipelineError(503, "Select part needs SAM 2, which this server cannot run")
    try:
        out = masker.prompt_session(job.id, (st.st_mtime_ns, st.st_size), lambda: imageio.load_image(p))
    except (ImportError, FileNotFoundError) as e:
        if getattr(job, "deleted", False):
            raise PipelineError(404, "the job was deleted") from e
        log.warning("job %s: SAM 2 unavailable for prompts (%s)", job.id, e)
        raise PipelineError(503, f"Select part needs SAM 2, which is not available here ({type(e).__name__})") from e
    except Exception as e:  # noqa: BLE001 - a CUDA OOM while embedding is a 503, the rest a clean 500
        raise _prompt_failed(job, e) from e
    # the deleted flag is set before the directory goes and the delete drops the sessions after it:
    # checked after the session is cached, a deleted job's embedding never stays behind
    if getattr(job, "deleted", False):
        _forget_prompts(job.id)
        raise PipelineError(404, "the job was deleted")
    return out


def _prompt_failed(job: Job, e: BaseException) -> PipelineError:
    """A prompt's model error as a PipelineError: 503 for a CUDA OOM (the card is shared)."""
    if _is_cuda_oom(e):
        log.warning("job %s: a prompt hit CUDA OOM", job.id)
        _free_cuda()
        return PipelineError(503, "The GPU has no memory to spare right now; try again in a moment",
                             retry_after=PROMPT_RETRY_AFTER_S)
    log.exception("job %s: a prompt failed", job.id)
    return PipelineError(500, f"segmentation failed: {_describe_error(e)}")


def _parse_prompt(job: Job, body: Any):
    w, h = _work_size(job)
    it = _interactive()
    try:
        return it.parse_prompt(body if body is not None else {}, w, h), (w, h)
    except it.PromptError as e:
        raise PipelineError(400, str(e)) from None


def _run_prompt(job: Job, prompt):
    """``(answer, session, computed)`` for a prompt, under the prompt GPU lock (the caller holds it)."""
    session, computed = _prompt_session(job)
    try:
        answer = _interactive().segment(session, prompt)
    except _interactive().PromptError as e:
        raise PipelineError(400, str(e)) from None
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001 - reported as a clean API error
        raise _prompt_failed(job, e) from e
    return answer, session, computed


def segment(job: Job, body: Any, cancelled: Optional[Callable[[], bool]] = None) -> dict[str, Any]:
    """``POST /api/jobs/{id}/segment``: SAM 2 prompted by the user's points (``[x, y, 1|0]``, work
    pixels, click order), box and ``multimask`` (`interactive.parse_prompt`). Returns the mask to
    show (`interactive.answer_json`: a 1-bit PNG data URL of its bounding box, the box, the area,
    SAM's score), the first step's alternatives, the ``pick`` and refinement ``crop`` a commit
    sends back, ``embed`` ('computed' the first time for this job, 'cached' after) and ``ms``. An
    empty prompt only prepares the embedding (``{"warm": true}``: the studio sends one when the
    tool opens) and, in the background, what a commit needs besides SAM (`_warm_commit`). 400 for
    a bad prompt, 409 before the analysis is done, 503 with a retry hint while the GPU is busy;
    ``cancelled()`` (the client went away) stops the wait for the GPU (`_prompt_gpu`)."""
    _require_ready(job)
    prompt, (w, h) = _parse_prompt(job, body)
    t0 = time.perf_counter()
    with _prompt_gpu("Select part", cancelled):
        if prompt.empty:
            session, computed = _prompt_session(job)
            _warm_commit(job)
            return {"warm": True, "embed": "computed" if computed else "cached", "size": [w, h],
                    "ms": round((time.perf_counter() - t0) * 1000, 1)}
        answer, session, computed = _run_prompt(job, prompt)
    out = _interactive().answer_json(answer, session.shape[1], session.shape[0])
    out.update(embed="computed" if computed else "cached", ms=round((time.perf_counter() - t0) * 1000, 1))
    return out


def _rank_detections(dets: list[dict], w: int, h: int, keep: int) -> list[dict]:
    """The detection boxes worth prompting, best first: finite, at least 4 px, at most 60 % of
    the image, a box overlapping a better one by IoU 0.5 (or 85 % inside it) dropped."""
    from .segmentation.smallparts import _box_inside, _box_iou
    good = []
    for d in dets or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in d["box"])
            score = float(d.get("score") if d.get("score") is not None else 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite([x0, y0, x1, y1, score]).all():
            continue
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
        if x1 - x0 < 4 or y1 - y0 < 4 or (x1 - x0) * (y1 - y0) > 0.6 * w * h:
            continue
        good.append({"box": [x0, y0, x1, y1], "score": score, "phrase": str(d.get("phrase") or d.get("label") or "")})
    good.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for d in good:
        if any(_box_iou(d["box"], k["box"]) > 0.5 or _box_inside(d["box"], k["box"]) >= 0.85 for k in kept):
            continue
        kept.append(d)
        if len(kept) >= keep:
            break
    return kept


def _find_rank(det_score: Optional[float], sam_score: float, on_object: float) -> float:
    """The order of Find part's candidates: the detector's score (square-rooted: OWLv2's scores of
    small parts are low and close together, 0.06-0.12 for the Corvette's caliper boxes) times
    SAM's predicted IoU squared (a clean mask of a part beats a speckled mask of half a wheel),
    halved for a mask mostly on the background."""
    det = 1.0 if det_score is None else max(0.0, float(det_score)) ** 0.5
    return det * max(0.0, float(sam_score)) ** 2 * (1.0 if on_object >= 0.5 else 0.5)


def _find_words(text: str) -> set[str]:
    """The words of a Find phrase or a group name, lower case, spellings folded (`FIND_SPELLINGS`),
    a plural's s dropped ("calipers" is "caliper"), a user kind's "user"/number left out."""
    import re as _re
    out: set[str] = set()
    for w in _re.findall(r"[a-z0-9]+", str(text).lower()):
        w = FIND_SPELLINGS.get(w, w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(FIND_SPELLINGS.get(w, w))
    return out


def _find_phrases(text: str) -> list[str]:
    """The detector phrases of Find part's ``text``: the text, its `FIND_SYNONYMS` (by word) and the
    prompts of every vocabulary part kind whose name holds every word of it ("caliper": "brake
    caliper"; "spring": "coil spring", "shock absorber"), at most `FIND_MAX_PHRASES`, no repeats."""
    phrases = [text]
    words = _find_words(text)
    for w in sorted(words):
        phrases.extend(FIND_SYNONYMS.get(w, ()))
    try:
        vocab = _smallparts().VOCAB
    except Exception:  # noqa: BLE001 - the synonyms are a bonus, never a failure of Find
        vocab = {}
    for kinds in vocab.values():
        for k in kinds:
            if words and words <= _find_words(f"{k.label} {k.plural} {k.key.replace('_', ' ')}"):
                phrases.extend(k.phrases)
    seen: list[str] = []
    for p in phrases:
        p = " ".join(str(p).lower().split())
        if p and p not in seen:
            seen.append(p)
    return seen[:FIND_MAX_PHRASES]


def _kind_phrases(kind: str) -> list[str]:
    """The detector prompts the vocabulary gives part kind ``kind`` in any object class ("exhaust":
    "exhaust muffler", "exhaust pipe"); [] for a user kind or without the vocabulary."""
    try:
        vocab = _smallparts().VOCAB
    except Exception:  # noqa: BLE001 - the phrases are a bonus, never a failure of Find
        return []
    return [p for kinds in vocab.values() for k in kinds if k.key == kind for p in k.phrases]


def _names_part(g: ColorGroup, words: set[str]) -> bool:
    """True when the part group ``g`` is what the Find phrase's ``words`` name: every word is a word
    of its name, its kind's label or plural, (a detected part) its kind or the head noun of one of
    the vocabulary's prompts of it, or a `FIND_SYNONYMS` word of one of those ("muffler" names
    "Exhaust", "saddle" names "Seat"); or the phrase is two or more words of one prompt ("shock
    absorber", "coil spring" name "Shock spring"; "motorcycle" alone names no "motorcycle seat")."""
    if not g.part or not words:
        return False
    names = f"{g.name} {g.part_label} {g.part_plural}"
    prompts: list[str] = []
    if not _userparts().is_user_kind(g.part):
        prompts = _kind_phrases(g.part)
        names += " " + g.part.replace("_", " ") + " " + " ".join(p.split()[-1] for p in prompts if p.split())
    known = _find_words(names)
    for w in list(known):
        for syn in FIND_SYNONYMS.get(w, ()):
            known |= _find_words(syn)
    if words <= known:
        return True
    return len(words) >= 2 and any(words <= _find_words(p) for p in prompts)


def find_parts(job: Job, body: Any, cancelled: Optional[Callable[[], bool]] = None) -> dict[str, Any]:
    """``POST /api/jobs/{id}/find``: the parts a phrase names ("spring", "brake caliper"). First the
    part groups the phrase names (`_names_part`: "caliper" names "Brake caliper" and a drawn "Far
    caliper"; at most `FIND_EXISTING_MAX`, ``existing`` true, no prompt: taking one selects that
    group), then OWLv2 on the work image and its four corner tiles for the phrase and its synonyms
    (`_find_phrases`; Florence-2's phrase grounding when OWLv2 is not available), the boxes ranked and
    de-duplicated, each prompted as a SAM box (`interactive.segment`, refined like a Select part box),
    up to `FIND_MAX` candidates in all, of distinct masks, ranked by `_find_rank`; a candidate that is
    already a part group of a kind the phrase does not name comes after the others. Each candidate
    carries its box, the detector's score, its mask, the part group it already is (``matches``: a
    part group it overlaps by IoU 0.5, with ``named``: whether the phrase names it; else null) and the
    ``prompt`` that commits it through ``groups/from_mask``. The groups and the group map are read in
    one edit generation. 503 while the GPU is busy; ``cancelled()`` (the client went away) stops the
    wait for it."""
    _require_ready(job)
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise PipelineError(400, "text must name a part, e.g. \"spring\"")
    text = " ".join(text.split())
    if len(text) > FIND_TEXT_MAX:
        raise PipelineError(400, f"text must be at most {FIND_TEXT_MAX} characters")
    w, h = _work_size(job)
    it = _interactive()
    t0 = time.perf_counter()
    detector = None
    phrases = _find_phrases(text)
    words = _find_words(text)

    def named() -> list[ColorGroup]:
        return sorted((g for g in job.groups() if not g.is_background and _names_part(g, words)),
                      key=lambda g: -g.area)[:FIND_EXISTING_MAX]

    room = FIND_MAX - len(named())                  # the part groups it names take their slots first
    found: list[tuple[dict, Any, Any]] = []
    with _prompt_gpu("Find part", cancelled):
        session, computed = _prompt_session(job)
        image = session.image
        try:
            dets = _partdetect().detect(image, phrases)
            detector = "owlv2" if dets is not None else None
            if dets is None:
                ground = getattr(_florence(), "ground", None)
                dets = ground(image, text) if callable(ground) else None
                detector = "florence" if dets is not None else None
        except Exception as e:  # noqa: BLE001 - reported as a clean API error
            raise _prompt_failed(job, e) from e
        t1 = time.perf_counter()
        for d in _rank_detections(dets or [], w, h, keep=3 * room):
            prompt = it.Prompt(box=tuple(d["box"]), multimask=True)
            try:
                answer = it.segment(session, prompt)
            except Exception as e:  # noqa: BLE001
                raise _prompt_failed(job, e) from e
            if answer.mask.area >= _userparts().MIN_PIECE_PX:
                found.append((d, answer, prompt))
    total = float(max(1, w * h))

    def build(layers: dict[str, Optional[np.ndarray]], groups: list[ColorGroup]) -> list[dict]:
        # the groups and the group map of one edit generation (`_snapshot`): an edit landing between
        # two reads pointed a candidate's ``matches`` and an existing mask at the wrong group
        gm = layers["group_map"]
        existing = sorted((g for g in groups if not g.is_background and _names_part(g, words)),
                          key=lambda g: -g.area)[:FIND_EXISTING_MAX]
        out: list[dict] = []
        shown: list[np.ndarray] = []
        for g in existing:
            m = gm == g.id
            if not m.any():
                continue
            mask = it.encode_mask(m)
            mask.update(score=None, index=0, refined=False, area_frac=round(mask["area"] / total, 6))
            out.append({"box": [float(v) for v in mask["bbox"]], "score": None, "rank": 1.0, "phrase": text,
                        "mask": mask, "matches": {"group_id": int(g.id), "name": g.name, "iou": 1.0, "named": True},
                        "prompt": None, "existing": True, "group_id": int(g.id)})
            shown.append(m)
        bg_ids = np.array([g.id for g in groups if g.is_background], np.int64)
        parts = {g.id: g for g in groups if g.part}
        ranked = []
        for d, answer, prompt in found:
            m = answer.mask.mask
            on_obj = 1.0 - (float(np.isin(gm[m], bg_ids).mean()) if bg_ids.size else 0.0)
            score = None if detector == "florence" else float(d["score"])
            # the part group it already is (IoU 0.5); one of a kind the phrase does not name goes
            # last (Find "spring": the detector's exhaust can, the Exhausts group, came before the spring)
            match = None
            ids, cnt = np.unique(gm[m], return_counts=True)
            n = int(m.sum())
            for gid, inter in zip(ids.tolist(), cnt.tolist()):
                g = parts.get(gid)
                if g is not None:
                    iou = inter / float(int(g.area) + n - inter)
                    if iou >= 0.5 and (match is None or iou > match["iou"]):
                        match = {"group_id": int(gid), "name": g.name, "iou": round(iou, 3),
                                 "named": _names_part(g, words)}
            other = match is not None and not match["named"]
            ranked.append((other, _find_rank(score, answer.mask.score, on_obj), d, answer, prompt, match))
        ranked.sort(key=lambda t: (t[0], -t[1]))
        room = FIND_MAX - len(out)
        kept: list[tuple] = []
        for item in ranked:
            m = item[3].mask.mask
            if any(it._iou(m, k[3].mask.mask) > FIND_DEDUP_IOU for k in kept) or \
                    any(it._iou(m, s) > FIND_DEDUP_IOU for s in shown):
                continue
            kept.append(item)
            if len(kept) >= room:
                break
        for _, rank, d, answer, prompt, match in kept:
            mask = it.answer_json(answer, w, h)["mask"]
            commit = dict(prompt.to_json(), pick=int(answer.pick),
                          crop=None if answer.crop is None else [int(v) for v in answer.crop])
            out.append({"box": [round(v, 1) for v in d["box"]],
                        "score": None if detector == "florence" else round(float(d["score"]), 4),
                        "rank": round(float(rank), 4), "phrase": d["phrase"] or text, "mask": mask,
                        "matches": match, "prompt": commit, "existing": False})
        return out

    out, _ = _snapshot(job, build)
    return {"text": text, "phrases": phrases, "detector": detector, "candidates": out,
            "embed": "computed" if computed else "cached",
            "detect_ms": round((t1 - t0) * 1000, 1), "ms": round((time.perf_counter() - t0) * 1000, 1)}


def _validate_part_name(name: Any) -> Optional[str]:
    if name is None:
        return None
    if not isinstance(name, str):
        raise PipelineError(400, "name must be a string")
    name = name.strip()
    if not name:
        return None
    if len(name) > NAME_MAX_LEN:
        raise PipelineError(400, f"name must be at most {NAME_MAX_LEN} characters")
    return name


def _user_kinds(job: Job, regions: list[Region]) -> list[str]:
    return [r.part_kind for r in regions if r.part_kind] + list(_load_user_parts(job))


#: The residual's glint reference per job (`materials.spec_reference`, about 30 ms on the whole
#: residual), keyed by the residual file's stamp: the residual never changes after the analysis.
_SPEC_Q: "OrderedDict[tuple, float]" = OrderedDict()
_SPEC_Q_LOCK = threading.Lock()
_warming: set[str] = set()


def _spec_q(job: Job, residual: np.ndarray) -> float:
    """The glint threshold's reference of ``job``'s residual (`materials.spec_reference`), cached."""
    try:
        st = os.stat(job.path("residual.npy"))
        key: Optional[tuple] = (job.id, st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None:
        with _SPEC_Q_LOCK:
            if key in _SPEC_Q:
                _SPEC_Q.move_to_end(key)
                return _SPEC_Q[key]
    from .segmentation import materials
    q = float(materials.spec_reference(residual))
    if key is not None:
        with _SPEC_Q_LOCK:
            _SPEC_Q[key] = q
            while len(_SPEC_Q) > 8:
                _SPEC_Q.popitem(last=False)
    return q


def _warm_commit(job: Job) -> None:
    """What a commit needs besides SAM, prepared in a background thread when Select part opens (the
    studio's empty prompt): the job's layers, the photo's hue and chroma the protect mask reads, the
    residual's glint reference, and the modules the carve, the region stats and the shine cues load
    on first use (scipy.ndimage alone took 30-40 ms). The first commit of a job paid 0.25 s more for
    them. CPU only; never fails a prompt."""
    if not _is_refined(job):
        return
    with _SPEC_Q_LOCK:
        if job.id in _warming:
            return
        _warming.add(job.id)

    def run() -> None:
        try:
            import scipy.ndimage  # noqa: F401 - loaded here, not in the first commit's region stats
            from .segmentation import materials  # noqa: F401
            _userparts()
            layers = load_layers(job)
            if layers.get("residual") is not None:
                _spec_q(job, layers["residual"])
            hue_chroma = getattr(_refine(), "_photo_hue_chroma", None)
            if callable(hue_chroma):
                hue_chroma(imageio.load_image(job.path("work.png")))
        except Exception:  # noqa: BLE001 - a warm-up only
            log.debug("job %s: warming the commit caches failed", job.id, exc_info=True)
        finally:
            with _SPEC_Q_LOCK:
                _warming.discard(job.id)

    threading.Thread(target=run, name=f"warm-commit-{job.id}", daemon=True).start()


def _shine_tags(work: np.ndarray, albedo: np.ndarray, residual: Optional[np.ndarray], labels: np.ndarray,
                regions: list[Region], ids: set[int], spec_q: Optional[float] = None) -> list[Region]:
    """``regions`` with the shininess cues (``shiny``, ``glint``, ``chrome``: the finish badge) of the
    part's regions ``ids`` measured on a window around them (`materials.shine_features` on the
    window, the glint threshold of the whole residual): every region of ``ids`` lies inside it,
    and the whole-image pass took 1-2.5 s. The other regions keep their cues. On any error the
    records are kept (the badge is advisory)."""
    try:
        from .segmentation import materials
        ids = sorted(int(i) for i in ids)
        m = np.isin(labels, ids)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            return regions
        H, W = labels.shape
        y0, y1 = max(0, int(ys.min()) - 4), min(H, int(ys.max()) + 5)
        x0, x1 = max(0, int(xs.min()) - 4), min(W, int(xs.max()) + 5)
        lut = np.zeros(int(labels.max()) + 1, np.int32)
        lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)          # 0: everything else in the window
        win = lut[labels[y0:y1, x0:x1]]
        q = spec_q if spec_q is not None else (materials.spec_reference(residual) if residual is not None else None)
        feats = materials.shine_features(win, albedo[y0:y1, x0:x1], work[y0:y1, x0:x1],
                                         None if residual is None else residual[y0:y1, x0:x1], spec_q=q)
        tmp = [Region(id=k + 1, area=int(feats.area[k + 1]), bbox=(0, 0, 1, 1), albedo_lab=(0.0, 0.0, 0.0),
                      albedo_hex="#000000", group_id=0, touches_border=False, source="user") for k in range(len(ids))]
        chrome = materials.chrome_regions(tmp, feats)
        cues = {rid: (round(float(feats.hl[k + 1]), 4), round(float(max(feats.clip[k + 1], feats.spec[k + 1])), 4),
                      (k + 1) in chrome) for k, rid in enumerate(ids)}
        return [replace(r, shiny=cues[r.id][0], glint=cues[r.id][1], chrome=cues[r.id][2]) if r.id in cues else r
                for r in regions]
    except Exception:  # noqa: BLE001 - an advisory badge, never a failure of the edit
        log.exception("shininess cues of a user part failed; keeping the records")
        return regions


def _seed_arrays(job: Job) -> Optional[dict[str, np.ndarray]]:
    """Every array of the job's regroup seed (`SEED_FILE`), or None without a readable one."""
    p = job.path(SEED_FILE)
    if not os.path.isfile(p):
        return None
    try:
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError) as e:
        log.warning("job %s: ignoring unreadable %s (%s)", job.id, SEED_FILE, e)
        return None


def _savez_fast(path: str, arrays: dict[str, np.ndarray]) -> None:
    """An ``.npz`` ``np.load`` reads, deflated at zlib level 1: the regroup seed took 24 ms at
    ``savez_compressed``'s level 6 in every commit, 10 ms here (158 kB instead of 84 kB for the Ducati)."""
    import zipfile
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for key, value in arrays.items():
            with zf.open(f"{key}.npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)


def _save_seed_arrays(job: Job, arrays: dict[str, np.ndarray]) -> None:
    tmp = job.path(f"{SEED_FILE}.{os.getpid()}-{threading.get_ident()}.tmp.npz")
    _savez_fast(tmp, arrays)
    os.replace(tmp, job.path(SEED_FILE))


def _seed_tags(arrays: dict[str, np.ndarray], n_in: int) -> dict[int, dict]:
    tags: dict[int, dict] = {}
    if "part_kind" not in arrays:
        return tags
    kinds, labs, plurals, inst = (arrays[k] for k in ("part_kind", "part_label", "part_plural", "part_instance"))
    for i, k in enumerate(kinds.tolist()[:n_in]):
        if k:
            tags[i] = {"kind": str(k), "label": str(labs[i]), "plural": str(plurals[i]), "instance": int(inst[i])}
    return tags


def _put_seed_tags(arrays: dict[str, np.ndarray], tags: dict[int, dict], n_in: int) -> None:
    arrays["part_kind"] = np.asarray([str(tags.get(i, {}).get("kind", "")) for i in range(n_in)], dtype=str)
    arrays["part_label"] = np.asarray([str(tags.get(i, {}).get("label", "")) for i in range(n_in)], dtype=str)
    arrays["part_plural"] = np.asarray([str(tags.get(i, {}).get("plural", "")) for i in range(n_in)], dtype=str)
    arrays["part_instance"] = np.asarray([int(tags.get(i, {}).get("instance", -1)) for i in range(n_in)], np.int32)


def _carve_seed(job: Job, arrays: dict[str, np.ndarray], mask: np.ndarray, final_part_ids: list[int],
                n_final: int, tag_rec: dict) -> Optional[dict[str, np.ndarray]]:
    """The regroup seed with the user part carved into its input label map too (`userparts.carve`:
    one input region of the part, tagged like a detected part, so a regroup clusters it apart),
    every per-input array extended and every final region of the part descending from it
    (``origin``). None when the seed does not fit the job (its label map or origin)."""
    up = _userparts()
    labels_in = np.asarray(arrays.get("labels"), np.int32) if "labels" in arrays else None
    origin = np.asarray(arrays.get("origin"), np.int32).ravel() if "origin" in arrays else None
    if labels_in is None or origin is None or labels_in.shape != mask.shape or labels_in.min() < 0:
        return None
    n_in = int(labels_in.max()) + 1
    c = up.carve(labels_in, mask)
    if c is None:
        return None
    n_in2 = int(c.labels.max()) + 1
    home = c.new_id if c.new_id is not None else c.part_ids[0]
    out = dict(arrays)
    out["labels"] = np.ascontiguousarray(c.labels, np.int32)
    if "bg" in arrays and arrays["bg"].size == n_in:
        bg = np.zeros(n_in2, np.int8)
        bg[:n_in] = np.asarray(arrays["bg"], np.int8)
        bg[c.part_ids] = 0                              # a part is the object
        out["bg"] = bg
    if "sources" in arrays and arrays["sources"].size == n_in:
        src = [str(v) for v in arrays["sources"]] + [up.USER_SOURCE] * (n_in2 - n_in)
        out["sources"] = np.asarray(src, dtype=str)
    tags = _seed_tags(arrays, n_in)
    for i in c.part_ids:
        tags[int(i)] = dict(tag_rec)
    _put_seed_tags(out, tags, n_in2)
    org = np.full(n_final, -1, np.int32)
    org[:min(len(origin), n_final)] = origin[:n_final]
    # the new region descends from the part's input region; a region wholly inside the part keeps
    # its origin (an input region the input carve covered too, or its final tag takes it there)
    for fid in final_part_ids:
        if int(fid) >= len(origin) or origin[int(fid)] < 0:
            org[int(fid)] = home
    out["origin"] = org
    return out


#: At most this many groups a commit is asked to take in whole (``take``: the pill's "Take all of").
TAKE_MAX = 8
#: Where a user part's carve is kept for Remove: ``parts/<kind>.npz`` holds, over the bounding box
#: of the region the carve made, the region each of its pixels came from (-1 elsewhere).
CARVE_DIR = "parts"


def _carve_path(job: Job, kind: str) -> str:
    return job.path(CARVE_DIR, f"{kind}.npz")


def _save_carve(job: Job, kind: str, labels_before: np.ndarray, new_region: np.ndarray) -> None:
    """Record where the pixels of a carve's new region (``new_region``: bool HxW) came from (their
    region in ``labels_before``), so Remove gives each back to its own region (atomic write)."""
    rows, cols = np.flatnonzero(new_region.any(1)), np.flatnonzero(new_region.any(0))
    if not rows.size:
        return
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    crop = np.where(new_region[y0:y1, x0:x1], labels_before[y0:y1, x0:x1], -1).astype(np.int32)
    os.makedirs(job.path(CARVE_DIR), exist_ok=True)
    tmp = job.path(CARVE_DIR, f"{kind}.{os.getpid()}-{threading.get_ident()}.tmp.npz")
    _savez_fast(tmp, {"bbox": np.array([x0, y0, x1, y1], np.int32), "donors": crop})
    os.replace(tmp, _carve_path(job, kind))


def _load_carve(job: Job, kind: str, shape: tuple[int, ...]) -> Optional[tuple[tuple[int, int, int, int], np.ndarray]]:
    """``((x0, y0, x1, y1), donors)`` of a user part's carve (`_save_carve`), None without a usable one."""
    p = _carve_path(job, kind)
    if not os.path.isfile(p):
        return None
    try:
        with np.load(p) as z:
            x0, y0, x1, y1 = (int(v) for v in z["bbox"])
            don = np.asarray(z["donors"], np.int32)
    except (OSError, ValueError, KeyError) as e:
        log.warning("job %s: ignoring unreadable %s (%s)", job.id, os.path.basename(p), e)
        return None
    H, W = shape[:2]
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H) or don.shape != (y1 - y0, x1 - x0):
        return None
    return (x0, y0, x1, y1), don


def _parse_take(body: Any) -> list[int]:
    """The ``take`` of a commit: distinct group ids (integers, at most `TAKE_MAX`), [] when absent."""
    raw = body.get("take") if isinstance(body, dict) else None
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > TAKE_MAX or \
            any(isinstance(v, bool) or not isinstance(v, int) or not (0 <= v < 2 ** 31) for v in raw):
        raise PipelineError(400, f"take must be a list of at most {TAKE_MAX} group ids")
    return sorted({int(v) for v in raw})


def add_user_part(job: Job, body: Any) -> dict[str, Any]:
    """``POST /api/jobs/{id}/groups/from_mask``: make the part a prompt selects a group of its own.

    The prompt (the same payload as `segment`, plus an optional ``name`` and ``take``: groups the
    selection overlaps that the part takes in whole, the pill's "Take all of") is run again here,
    the client's mask is never used; it runs under the GPU lock alone, before the job's edit lock
    is taken, so an edit of the job never waits behind a commit that waits for the GPU. A part
    instance the mask covers almost whole is taken in whole (`userparts.absorb_parts`, measured over
    the instance's regions together: no sliver of it stays behind). The mask is carved into the
    label map (`userparts.carve`: region ids stay, a region wholly inside the part joins it, the
    pieces cut off the others become one new region of source 'user', no speck is made, the part's
    holes stay), the part's regions are tagged like a detected part (kind ``user_<n>``, label the
    name, else the name of the group it replaces, else "Part <n>") and get a group of their own, so
    the colour clustering never takes it, the automatic rules never flag it background, it is never
    pruned as junk and a regroup gives it back (the regroup seed carries it as an input region of
    its own, ``user_flags.json`` lists it). No other group changes its name (a part group's
    automatic kind name follows its instance count: "Brake calipers" that lost one is "Brake
    caliper"), its lock, its background flag or its paint; the new part is unpainted unless it
    replaces a group (a part group or a colour group taken in whole that makes up most of it), whose
    paint it keeps. Every group taken in whole is recorded (its regions, name, flags and paint) and a
    region of a group only partly taken records a region of that group left outside, so Remove gives
    each back where it was; a user part taken in whole waits inside the new one. The protect mask is
    recomputed (the islands are pixel facts and stay), every file is written atomically inside the
    job's edit lock and generation, and the caches are dropped. Returns the job with
    ``created_group`` (the new group's id) and ``created_part`` (its kind, name, area, regions,
    ``took_in``: the groups it took in whole, ``replaced``: the one it replaces or None, and the
    timings)."""
    name = _validate_part_name(body.get("name") if isinstance(body, dict) else None)
    take = _parse_take(body)
    _require_ready(job)
    prompt, (w, h) = _parse_prompt(job, body)
    if prompt.empty:
        raise PipelineError(400, "click on the part or draw a box around it")
    up = _userparts()
    grouping = _grouping()
    t0 = time.perf_counter()
    tm: dict[str, float] = {}

    def mark(what: str) -> None:
        tm[what] = round((time.perf_counter() - t0) * 1000, 1)

    with _prompt_gpu():
        answer, session, _ = _run_prompt(job, prompt)
    mark("sam")
    with job.edit_lock:
        _require_ready(job)
        mark("lock")
        layers = load_layers(job)
        labels0 = layers["labels"]
        regions = _load_regions(job)
        groups = job.groups()
        by_id = {r.id: r for r in regions}
        by_gid = {g.id: g for g in groups}
        # the work image the prompt session decoded (keyed by work.png's stamp): no second decode
        work = np.ascontiguousarray(session.image) if getattr(session, "image", None) is not None \
            else imageio.load_image(job.path("work.png"))
        albedo = layers["albedo"]
        mark("load")
        mask = np.asarray(answer.mask.mask, bool)
        if mask.shape != labels0.shape:
            raise PipelineError(409, "the working image and the label map disagree; analyse the photo again")
        if take:
            gm0 = layers["group_map"]
            for gid in take:
                if gid not in by_gid:
                    raise PipelineError(400, f"take: unknown group {gid}")
                if not (mask & (gm0 == gid)).any():
                    raise PipelineError(400, f"take: {by_gid[gid].name} is not under the selection")
            mask = mask | np.isin(gm0, take)
        mask, _ = up.absorb_parts(mask, labels0, regions)
        c = up.carve(labels0, mask)
        if c is None or c.area < up.MIN_PIECE_PX:
            raise PipelineError(400, "The selection is too small to become a part; click on the part again")
        if c.new_id is None and len(c.covered) == len(regions):
            raise PipelineError(400, "The selection covers the whole image; leave something out (Shift-click)")
        covered = set(c.covered)
        # every group it takes in whole (a part group or a colour group: the Ducati's gold "748"
        # inside a drawn caliper), and the one it replaces (most of the new part)
        took = [g for g in groups if g.region_ids and set(g.region_ids) <= covered]
        took.sort(key=lambda g: (-g.area, g.id))
        took_ids = {g.id for g in took}
        replaced = next((g for g in took if g.area >= up.REPLACE_SHARE * c.area), None)
        kind, number = up.next_user_kind(_user_kinds(job, regions))
        label = name or (replaced.name if replaced is not None else up.default_label(number))
        labels = c.labels
        mark("carve")
        affected = set(c.part_ids) | set(c.donors)
        stats = up.region_stats(labels, albedo, affected)
        out: list[Region] = []
        for r in regions:
            s = stats.get(r.id)
            if s is not None:
                r = replace(r, area=s["area"], bbox=s["bbox"], albedo_lab=s["lab"],
                            albedo_hex=imageio.lab_to_hex(s["lab"]), touches_border=s["touches_border"])
            if r.id in c.part_ids:
                r = up.tag(r, kind, label, label, 0)
            out.append(r)
        if c.new_id is not None:
            s = stats[c.new_id]
            parent = by_id[max(c.donors, key=lambda k: (c.donors[k], -k))] if c.donors else None
            out.append(up.tag(Region(
                id=c.new_id, area=s["area"], bbox=s["bbox"], albedo_lab=s["lab"],
                albedo_hex=imageio.lab_to_hex(s["lab"]), group_id=parent.group_id if parent is not None else 0,
                touches_border=s["touches_border"], source=up.USER_SOURCE,
                confidence=round(float(answer.mask.score), 4), shiny=parent.shiny if parent else 0.0,
                glint=parent.glint if parent else 0.0), kind, label, label, 0))
        mark("stats")
        # the shininess cues of the part (on a window around it; the regions it cut keep theirs)
        residual = layers.get("residual")
        out = _shine_tags(work, albedo, residual, labels, out, set(c.part_ids),
                          spec_q=_spec_q(job, residual) if residual is not None else None)
        mark("shine")
        donor_groups = {by_id[r].group_id for r in affected if r in by_id}
        recompute = {g.id for g in groups if g.part and g.id in donor_groups}
        fresh = max([g.id for g in groups] + [-1]) + 1
        assignment = {r.id: (fresh if r.id in c.part_ids else r.group_id) for r in out}
        carry = {g.id: {"name": grouping._keep_name(g) if g.id in recompute else g.name, "locked": g.locked,
                        "is_background": g.is_background, "ref_lab": None if g.id in donor_groups else g.ref_lab}
                 for g in groups}
        carry[fresh] = {"name": None, "locked": False, "is_background": False, "ref_lab": None}
        old_groups = list(groups)
        # every other group keeps its flags as they are (carry): a detected part the user flagged
        # background stayed flagged; the new part is unlocked and never background
        regions, groups, group_map = grouping._finalize(out, labels, assignment, carry, sort_by_area=False)
        new_of = {p: i for i, p in enumerate(sorted(set(assignment.values())))}
        up.keep_names(old_groups, groups, new_of, recompute)
        grouping._check_state(regions, groups)
        created = new_of[fresh]
        mark("groups")
        protect = None
        if _is_refined(job):
            *_, protect = _refine().refine_after_edit(work, albedo, labels, regions, groups, group_map,
                                                      layers.get("islands"), regrouped=False)
        mark("protect")
        seed = _seed_arrays(job)
        new_seed = None
        if seed is not None:
            new_seed = _carve_seed(job, seed, c.mask, c.part_ids, int(labels.max()) + 1,
                                   {"kind": kind, "label": label, "plural": label, "instance": 0})
            if new_seed is None:
                log.warning("job %s: the regroup seed does not fit the label map; the user part relies on its tags", job.id)
        parts = _load_user_parts(job)
        # a user part that loses regions to this one lists them no more; one taken in whole waits
        # inside this one (`_sync_user_parts`) and comes back when this one is removed
        for k, rec in parts.items():
            regs = [int(v) for v in rec.get("regions") or []]
            keep = [v for v in regs if v not in covered]
            if len(keep) != len(regs):
                rec["regions"] = keep
                if not keep:
                    rec["inside"] = kind
        # where the pixels came from: Remove puts the part back into the group of its main donor,
        # gives the regions it covered their tags (``was``), every group it took in whole back as it
        # was (``groups``: regions, name, flags, paint) and each region of a group it took only in
        # part back to the group holding a region of it left outside (``homes``)
        was = {str(int(rid)): ({"kind": by_id[rid].part_kind, "label": by_id[rid].part_label,
                                "plural": by_id[rid].part_plural, "instance": int(by_id[rid].part_instance)}
                               if by_id[rid].part_kind else None) for rid in c.covered if rid in by_id}
        homes: dict[str, int] = {}
        for rid in c.covered:
            g = by_gid.get(by_id[rid].group_id) if rid in by_id else None
            if g is None or g.id in took_ids:
                continue
            rest = [x for x in g.region_ids if x not in covered and x in by_id]
            if rest:
                homes[str(int(rid))] = int(max(rest, key=lambda x: (by_id[x].area, -x)))
        took_rec = [{"regions": sorted(int(v) for v in g.region_ids), "name": g.name, "locked": bool(g.locked),
                     "is_background": bool(g.is_background), "part": g.part or ""} for g in took]
        took_paint: dict[int, str] = {}

        def remap(current: dict[str, Any]) -> dict[str, Any]:
            # every group keeps its own paint; the new part is unpainted unless it replaces a group
            for g in took:
                if current.get(str(g.id)):
                    took_paint[g.id] = current[str(g.id)]
            new = _paint_by_id(current, new_of, skip={fresh})
            if replaced is not None and current.get(str(replaced.id)):
                new[str(created)] = current[str(replaced.id)]
            return new

        mark("seed")
        with _writing(job):
            if protect is not None:
                _write_masks(job, None, protect)
            _write_labels(job, labels, work, display=False)
            if c.new_id is not None:
                _save_carve(job, kind, labels0, labels == c.new_id)
            mark("write_labels")
            if new_seed is not None:
                _save_seed_arrays(job, new_seed)
            mark("write_seed")
            _write_grouping(job, list(regions), list(groups), np.asarray(group_map), remap=remap, display=False)
            mark("write_groups")
            for rec_g, g in zip(took_rec, took):
                if took_paint.get(g.id):
                    rec_g["paint"] = took_paint[g.id]
            parts[kind] = {"label": label, "regions": [int(v) for v in c.part_ids],
                           "donors": {str(int(k)): int(v) for k, v in c.donors.items()}, "was": was,
                           "groups": took_rec, "homes": homes}
            if c.new_id is not None:
                parts[kind]["new_id"] = int(c.new_id)
            _save_user_parts(job, parts)
        _install_layers(job, layers, labels, np.asarray(group_map), protect)
        mark("write")
        log.info("job %s: user part %s %r, %d px in regions %s (%d new), group %s, took in %s, ms %s", job.id, kind,
                 label, c.area, c.part_ids, 0 if c.new_id is None else 1, created, [g.name for g in took], tm)
        snap = job.snapshot()
    snap["created_group"] = created
    snap["created_part"] = {"kind": kind, "name": label, "area": c.area, "regions": [int(v) for v in c.part_ids],
                            "refined": bool(answer.mask.refined), "took_in": [g.name for g in took],
                            "replaced": replaced.name if replaced is not None else None, "ms": tm}
    return snap


def _follow_user_parts(regions: list[Region], groups: list[ColorGroup], group_map: np.ndarray, labels: np.ndarray):
    """After a merge, split or move: `userparts.normalize` when the job has user parts."""
    up = _userparts()
    if not any(up.is_user_kind(r.part_kind) for r in regions) and not any(up.is_user_kind(g.part) for g in groups):
        return regions, groups, group_map, False
    return up.normalize(regions, groups, labels)


def _join_target_tags(regions: list[Region], groups: list[ColorGroup], target: int, moving,
                      dissolve: bool = False) -> list[Region]:
    """The part tags of the regions an edit sends into group ``target`` (``moving(region)``: the
    regions of the groups merged into it, or the regions moved there), so the result is what the
    user asked for whatever the sizes: into a user part, a colour region or another user part
    joins it; into a detected part, a user part becomes a new instance of it (a far caliper into
    "Brake caliper"); into a colour group, a user part is dissolved. With ``dissolve`` (Remove
    part) a user part is dissolved into any target, a detected part included. Detected parts' own
    regions keep their tags."""
    up = _userparts()
    tg = next((g for g in groups if g.id == target), None)
    kind = tg.part if tg is not None else ""
    label = (tg.part_label or kind) if tg is not None else ""
    plural = (tg.part_plural or label) if tg is not None else ""
    inst = max([r.part_instance for r in regions if kind and r.part_kind == kind] + [-1])
    first = min([max(0, r.part_instance) for r in regions if kind and r.part_kind == kind] + [0])
    new_inst: dict[str, int] = {}
    out = []
    for r in regions:
        if moving(r) and r.group_id != target:
            user = up.is_user_kind(r.part_kind)
            if dissolve:
                if user:
                    r = up.untag(r)
            elif up.is_user_kind(kind):
                if not r.part_kind or (user and r.part_kind != kind):
                    r = up.tag(r, kind, label, plural, first)
            elif kind:
                if user:
                    if r.part_kind not in new_inst:
                        inst += 1
                        new_inst[r.part_kind] = inst
                    r = up.tag(r, kind, label, plural, new_inst[r.part_kind])
            elif user:
                r = up.untag(r)
        out.append(r)
    return out


def _remove_target(job: Job, regions: list[Region], groups: list[ColorGroup], group_map: np.ndarray,
                   part: ColorGroup) -> Optional[int]:
    """The group a removed user part goes back into: the group now holding the region that gave
    it the most pixels (the registry's ``donors``), else the group owning most of a 3 px ring
    around it; None when there is neither."""
    rec = _load_user_parts(job).get(part.part) or {}
    g_of = {r.id: r.group_id for r in regions}
    votes: Counter[int] = Counter()
    for rid, px in (rec.get("donors") or {}).items():
        gid = g_of.get(int(rid))
        if gid is not None and gid != part.id:
            votes[gid] += int(px)
    if votes:
        return votes.most_common(1)[0][0]
    import cv2
    m = (np.asarray(group_map) == part.id).astype(np.uint8)
    if not m.any():
        return None
    ring = cv2.dilate(m, np.ones((7, 7), np.uint8)).astype(bool) & ~m.astype(bool)
    ids, cnt = np.unique(np.asarray(group_map)[ring], return_counts=True)
    keep = [(int(c), int(i)) for i, c in zip(ids.tolist(), cnt.tolist()) if i != part.id]
    return max(keep)[1] if keep else None


def _undo_carve(job: Job, rec: dict, part: ColorGroup, regions: list[Region], labels: np.ndarray,
                albedo: np.ndarray) -> tuple[list[Region], Optional[int], Optional[int]]:
    """In place on ``labels`` (a copy the caller owns): the pixels of the region the part's carve made
    (``new_id``) go back to the regions they came from (`_save_carve`), except those of the region
    that gave the most of them, which keep the new region's id (a region is never emptied, so no id
    is renumbered) and become that region's twin: its tag, its shine cues. Returns ``(regions with
    their stats refreshed, the new region's id, its main donor)``, or ``(regions, None, None)`` when
    the part has no carve record, no new region any more, or it holds no pixel of it."""
    n_id = rec.get("new_id")
    undo = _load_carve(job, part.part, labels.shape) if n_id is not None else None
    by_id = {r.id: r for r in regions}
    if undo is None or n_id not in by_id or by_id[n_id].group_id != part.id:
        return regions, None, None
    (x0, y0, x1, y1), don = undo
    view = labels[y0:y1, x0:x1]
    sel = (view == n_id) & (don >= 0) & (don < len(regions))
    if not sel.any():
        return regions, None, None
    ids, cnt = np.unique(don[sel], return_counts=True)
    main = int(ids[np.argmax(cnt)])
    if main not in by_id or main == n_id:
        return regions, None, None
    back = sel & (don != main)
    view[back] = don[back]
    stats = _userparts().region_stats(labels, albedo, {int(n_id), *(int(v) for v in ids)})
    src = by_id[main]
    out = []
    for r in regions:
        s = stats.get(r.id)
        if s is not None:
            r = replace(r, area=s["area"], bbox=s["bbox"], albedo_lab=s["lab"], albedo_hex=imageio.lab_to_hex(s["lab"]),
                        touches_border=s["touches_border"])
        if r.id == n_id:
            r = replace(r, part_kind=src.part_kind, part_label=src.part_label, part_plural=src.part_plural,
                        part_instance=src.part_instance, shiny=src.shiny, glint=src.glint, chrome=src.chrome,
                        backdrop=src.backdrop)
        out.append(r)
    return out, int(n_id), main


def _remove_user_part(job: Job, regions: list[Region], groups: list[ColorGroup], group_map: np.ndarray,
                      labels: np.ndarray, part: ColorGroup, albedo: Optional[np.ndarray] = None):
    """Remove part (``merge {group_ids: [part], dissolve: true}``): the part's regions go back where
    their pixels came from. The region the carve made gives every pixel back to the region it came
    from (`_undo_carve`: ``labels``, which the caller copied, is changed; the pixels of the region
    that gave the most stay in the new region, a twin of it in its group), so each group has its
    pixels again (a part cut out of a drawn part and the tyre gave that drawn part the whole cut, and
    as the colour region it made the most of, dissolved it). Every group the part had taken in whole
    (the registry's ``groups``: a colour group, a detected part, an earlier user part, one of a part's
    split instances) is a group of its own again, with the regions of it the part still holds, its
    name, its lock and background flags and its paint of then; a region of a group the part took only
    in part goes back to the group now holding a region of that group left outside (``homes``, when
    that group fits it: of its part kind for a part's region, no user part for a colour region); a
    region wholly inside the part keeps the part tag it had (``was``: a detected part is that part
    again). Without a carve record (an older job) the pieces the part cut off the other regions go to
    the group now holding the region that gave the most of them (`_remove_target`), tagged like it
    when it is a detected part, and a record without ``groups`` sends a detected part's region to its
    kind's group (a fresh one when the kind has none left). Every other group keeps its id order,
    flags and name (`userparts.keep_names`). Returns ``(regions, groups, group_map, remap, undo)``:
    ``remap`` makes the mapping after the edit, in which every group keeps its own paint, the part's
    paint goes with the part (carried by membership it painted the whole group it went back to), and
    every group given back gets the paint it had then; ``undo`` is ``{"labels": whether the label map
    changed, "origin": {new region: its main donor}}`` for the regroup seed, with ``restored`` (the
    names of the groups given back) and ``home`` (the group the carve's region went back to)."""
    grouping = _grouping()
    up = _userparts()
    rec = _load_user_parts(job).get(part.part) or {}
    regions = _restore_tags(job, regions, part)
    undo: dict[str, Any] = {"labels": False, "origin": {}}
    n_id = main = None
    if albedo is not None:
        regions, n_id, main = _undo_carve(job, rec, part, regions, labels, albedo)
    g_of = {r.id: r.group_id for r in regions}
    home = None
    if n_id is not None:
        undo = {"labels": True, "origin": {n_id: main}}
        if g_of.get(main) not in (None, part.id):
            home = g_of[main]                           # the twin of its main donor joins that one's group
    if home is None:
        home = _remove_target(job, regions, groups, group_map, part)
    if home is None:
        raise PipelineError(400, "there is no group around this part to put it back into")
    by_gid = {g.id: g for g in groups}
    mine = {r.id for r in regions if r.group_id == part.id}
    fresh = max([g.id for g in groups] + [-1]) + 1
    assignment = {r.id: r.group_id for r in regions}
    gaining: set[int] = {home}
    # the groups it had taken in whole: one fresh group each, as they were
    back: list[tuple[int, dict]] = []
    placed: set[int] = set()
    for gr in rec.get("groups") or []:
        ids = [int(v) for v in gr.get("regions") or [] if int(v) in mine and int(v) not in placed]
        if not ids:
            continue
        gid = fresh + len(back)
        back.append((gid, gr))
        for rid in ids:
            assignment[rid] = gid
            placed.add(rid)
    if n_id is not None and n_id in mine and n_id not in placed:
        assignment[n_id] = home                         # its main donor's twin, in that one's group
        placed.add(n_id)
    kind_group = {g.part: g.id for g in groups if g.part and g.id != part.id}
    new_groups: dict[str, int] = {}
    homes = rec.get("homes") or {}
    for r in regions:
        if r.id not in mine or r.id in placed:
            continue
        anchor = homes.get(str(r.id))
        gid = g_of.get(int(anchor)) if anchor is not None else None
        if gid is not None and gid != part.id and gid in by_gid:
            there = by_gid[gid]
            if (there.part == r.part_kind) if r.part_kind else not up.is_user_kind(there.part):
                assignment[r.id] = gid
                gaining.add(gid)
                continue
        if r.part_kind and r.part_kind != by_gid[home].part:
            gid = kind_group.get(r.part_kind)
            if gid is None:
                gid = new_groups.setdefault(r.part_kind, fresh + len(back) + len(new_groups))
            assignment[r.id] = gid
            gaining.add(gid)
        else:
            assignment[r.id] = home
    recompute = {g.id for g in groups if g.part and g.id in gaining}
    carry = {g.id: {"name": grouping._keep_name(g) if g.id in recompute else g.name, "locked": g.locked,
                    "is_background": g.is_background, "ref_lab": None if g.id in gaining else g.ref_lab}
             for g in groups}
    for gid, gr in back:
        carry[gid] = {"name": str(gr.get("name") or "") or None, "locked": bool(gr.get("locked")),
                      "is_background": bool(gr.get("is_background")), "ref_lab": None}
    for gid in new_groups.values():
        carry[gid] = {"name": None, "locked": False, "is_background": False, "ref_lab": None}
    old_groups = list(groups)
    regions, groups, group_map = grouping._finalize(regions, labels, assignment, carry, sort_by_area=False)
    new_of = {p: i for i, p in enumerate(sorted(set(assignment.values())))}
    up.keep_names(old_groups, groups, new_of, recompute,
                  restore={new_of[gid]: str(gr.get("name") or "") for gid, gr in back})
    grouping._check_state(regions, groups)
    undo["restored"] = [groups[new_of[gid]].name for gid, _ in back]
    undo["home"] = groups[new_of[home]].name if home in new_of else None
    back_paint = {new_of[gid]: str(gr["paint"]) for gid, gr in back if gr.get("paint")}
    paints = rec.get("paints") or {}                   # an older record: the paint by part kind
    legacy = {new_of[gid]: kind for kind, gid in new_groups.items()}

    def remap(current: dict[str, Any]) -> dict[str, Any]:
        out = _paint_by_id(current, new_of, skip={part.id})
        for gid, paint in back_paint.items():
            out.setdefault(str(gid), paint)
        for gid, kind in legacy.items():
            if paints.get(kind):
                out.setdefault(str(gid), paints[kind])
        return out

    return regions, groups, group_map, remap, undo


def _paint_by_id(mapping: dict[str, Any], new_of: dict[int, int], skip: Iterable[int] = ()) -> dict[str, Any]:
    """``mapping`` re-keyed by group id for an edit that knows where each group went (``new_of``:
    old id -> new id): every group still there keeps its own paint, the others' (and ``skip``'s) go."""
    skip = {int(i) for i in skip}
    out: dict[str, Any] = {}
    for k, v in (mapping or {}).items():
        try:
            old = int(k)
        except (TypeError, ValueError):
            continue
        if old in new_of and old not in skip:
            out[str(new_of[old])] = v
    return out


def _restore_tags(job: Job, regions: list[Region], part: ColorGroup) -> list[Region]:
    """The part tags a removed user part's regions go back to: a region wholly inside the part keeps
    the tag it had before (``was`` in the registry: a detected part it covered), the pieces it cut
    off the others take the tag of the region that gave the most of them (a far caliper cut out of
    the "Brake disc" is the disc again), and anything else is a colour region again."""
    up = _userparts()
    rec = _load_user_parts(job).get(part.part) or {}
    by_id = {r.id: r for r in regions}
    donors = rec.get("donors") or {}
    main = by_id.get(int(max(donors, key=lambda k: (donors[k], -int(k))))) if donors else None
    donor_tag = None
    if main is not None and main.part_kind and not up.is_user_kind(main.part_kind):
        donor_tag = (main.part_kind, main.part_label, main.part_plural, max(0, main.part_instance))
    was = rec.get("was") or {}
    out = []
    for r in regions:
        if r.part_kind == part.part:
            t = was.get(str(r.id), "absent")
            if isinstance(t, dict) and t.get("kind"):
                r = up.tag(r, t["kind"], t.get("label", ""), t.get("plural", ""), int(t.get("instance", 0)))
            elif t == "absent" and r.source == up.USER_SOURCE and donor_tag is not None:
                r = up.tag(r, *donor_tag)
            else:
                r = up.untag(r)
        out.append(r)
    return out


def _sync_user_parts(job: Job, regions: list[Region], origin_fix: Optional[dict[int, int]] = None) -> None:
    """After an edit that moved user parts' tags: the registry follows the regions, the carve records
    of the parts that are gone are deleted, and the regroup seed's input tags follow their
    descendants (`userparts.sync_seed_tags`); ``origin_fix`` (final region -> the region whose input
    region it now descends from: the twin Remove part left of a carve's main donor) re-points the
    seed's ``origin``, so a regroup gives the twin its donor's group. Called while writing."""
    up = _userparts()
    old = _load_user_parts(job)
    now = up.registry_of(regions)
    for kind, rec in now.items():
        if kind in old:
            rec["label"] = old[kind].get("label") or rec["label"]
            for key in USER_PART_HISTORY:
                if old[kind].get(key) is not None and old[kind].get(key) != {} and old[kind].get(key) != []:
                    val = old[kind][key]
                    rec[key] = [dict(v) for v in val] if isinstance(val, list) else dict(val) if isinstance(val, dict) else val
    # a user part another one took in whole waits inside it (no regions of its own): kept while that
    # one is there (or waits itself), back with its tags when that one is removed, gone with it
    grew = True
    while grew:
        grew = False
        for kind, rec in old.items():
            if kind not in now and rec.get("inside") in now:
                now[kind] = dict(rec, regions=[])
                grew = True
    if now != old:
        _save_user_parts(job, now)
    try:
        for f in os.listdir(job.path(CARVE_DIR)):
            if f.endswith(".npz") and ".tmp" not in f and f[:-4] not in now:
                os.remove(job.path(CARVE_DIR, f))
    except FileNotFoundError:
        pass
    arrays = _seed_arrays(job)
    if arrays is None or "labels" not in arrays or "origin" not in arrays:
        return
    n_in = int(np.asarray(arrays["labels"]).max()) + 1
    moved = False
    if origin_fix:
        origin = np.asarray(arrays["origin"], np.int32).ravel().copy()
        for rid, src in origin_fix.items():
            if 0 <= int(rid) < len(origin) and 0 <= int(src) < len(origin) and origin[int(rid)] != origin[int(src)]:
                origin[int(rid)] = origin[int(src)]
                moved = True
        arrays["origin"] = origin
    tags, changed = up.sync_seed_tags(_seed_tags(arrays, n_in), arrays["origin"], regions, n_in)
    if changed:
        _put_seed_tags(arrays, tags, n_in)
    if changed or moved:
        _save_seed_arrays(job, arrays)


# =============================================================================
# convenience for the API layer
# =============================================================================

def parse_analysis_options(fields: dict[str, Any]) -> AnalysisOptions:
    """Validate `detail`, `intrinsic`, `max_groups`, `delta_e` from a form or JSON
    body (strings accepted for the numbers). PipelineError(400) on bad values."""
    opts = AnalysisOptions()
    detail = fields.get("detail")
    if detail not in (None, ""):
        if detail not in DETAIL_LEVELS:
            raise PipelineError(400, f"detail must be one of {list(DETAIL_LEVELS)}")
        opts.detail = detail
    intrinsic = fields.get("intrinsic")
    if intrinsic not in (None, ""):
        if intrinsic not in INTRINSIC_METHODS:
            raise PipelineError(400, f"intrinsic must be one of {list(INTRINSIC_METHODS)}")
        opts.intrinsic = intrinsic
    mg = fields.get("max_groups")
    if mg not in (None, "", "null", "auto"):
        opts.max_groups = _bounded_int(mg, "max_groups", 1, MAX_GROUPS_LIMIT)
    de = fields.get("delta_e")
    if de not in (None, ""):
        opts.delta_e = _bounded_float(de, "delta_e", DELTA_E_MIN, DELTA_E_MAX)
    return opts


def validate_state(job: Job, body: dict[str, Any]) -> dict[str, Any]:
    """Validated `{mapping?, render_options?, palette_id?, ignore_background?}` for `PUT /state`."""
    out: dict[str, Any] = {}
    if "mapping" in body:
        out["mapping"] = mapping_to_json(_validate_mapping(job, body["mapping"]))
    if "render_options" in body:
        out["render_options"] = _validate_options(body["render_options"]).to_dict()
    if "palette_id" in body:
        pid = body["palette_id"]
        if pid is not None and (not isinstance(pid, str) or not pid.isalnum() or len(pid) > 64):
            raise PipelineError(400, "palette_id must be an alphanumeric id or null")
        out["palette_id"] = pid
    if "ignore_background" in body:
        v = body["ignore_background"]
        if not isinstance(v, bool):
            raise PipelineError(400, "ignore_background must be true or false")
        out["ignore_background"] = v
    return out


__all__ = [
    "PipelineError", "analyze", "apply_group_edit", "decode_region_ids", "effective_groups", "encode_group_ids",
    "encode_region_ids", "enqueue", "export", "get_renderer", "gpu_lock", "ignores_background", "invalidate",
    "load_layers", "model_status", "parse_analysis_options", "queue_length", "refresh_panel_view",
    "remap_mapping", "render_preview", "resume_pending", "save_state", "start_warmup", "start_worker",
    "validate_state",
]
