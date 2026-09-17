"""Jobs: one uploaded image plus everything derived from it, and the on-disk registry.

A `Job` owns a directory under `config.JOBS_DIR` (see docs/ARCHITECTURE.md §3.6 for the
artifact layout) and a `meta` dict that is *exactly* the JSON returned by
`GET /api/jobs/{id}`. Mutations go through the job's lock and are persisted with
`save()`; progress is broadcast to SSE subscribers through `publish()`.

Nothing here touches the GPU or the models; the analysis itself lives in
`recolor.pipeline`.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import queue
import secrets
import shutil
import threading
import time
from typing import Any, Optional

import numpy as np

from . import config, imageio
from .types import STAGES, AnalysisOptions, ColorGroup, ImageInfo, StageState

STATUSES = ("queued", "analyzing", "ready", "error")
STAGE_STATES = ("idle", "running", "done", "error", "skipped")

# Subscriber queues are bounded so a stalled client can never make a publisher block;
# a full queue drops the oldest event (the client re-syncs from the replay on reconnect).
_SUBSCRIBER_QUEUE_SIZE = 512

log = logging.getLogger("recolor.jobs")


def _now() -> float:
    return time.time()


def new_job_id() -> str:
    """12 hex characters, unguessable enough for share links, safe as a directory name."""
    return secrets.token_hex(6)


def is_job_id(value: str) -> bool:
    """True when `value` has the shape produced by `new_job_id` (used to reject path tricks)."""
    return isinstance(value, str) and len(value) == 12 and all(c in "0123456789abcdef" for c in value)


def _original_filename(name: str) -> str:
    """`original.png` when the upload was a PNG (lossless stays lossless), else `original.jpg`."""
    return "original.png" if name.lower().endswith(".png") else "original.jpg"


class Job:
    """One uploaded image and everything derived from it.

    Guarantees:
    - `meta` always has every key of the API `Job` object (see docs/ARCHITECTURE.md §3.7).
    - Every mutation helper (`set_stage`, `set_status`, `set_groups`, `update`) is
      thread-safe and publishes the matching SSE event.
    - `save()` writes `job.json` atomically (unique temp file + rename, under the job
      lock), so a crash mid-write never leaves a truncated file behind and concurrent
      savers (API handlers racing the worker) never trip over each other's temp file.
    - Once `deleted` is set (by `JobRegistry.delete`) nothing is written to disk any
      more, so a worker still running a stage cannot resurrect the job directory.
    """

    def __init__(self, id: str, dir: str, meta: dict[str, Any]):
        self.id = id
        self.dir = dir
        self.meta = meta
        self.lock = threading.RLock()
        self._subscribers: list[queue.Queue] = []
        self._stage_started: dict[str, float] = {}
        self._last_tick: dict[str, tuple[float, float]] = {}
        # The freshly uploaded array is kept in memory until ingest has written the
        # working-resolution files; it is never re-read from here after that.
        self.pending_image: Optional[np.ndarray] = None
        # Set by JobRegistry.delete() under the lock; every persist becomes a no-op.
        self.deleted = False

    # ------------------------------------------------------------------ construction

    @staticmethod
    def new_meta(id: str, name: str, image: ImageInfo, options: AnalysisOptions) -> dict[str, Any]:
        """The initial `queued` meta record with every stage idle."""
        return {
            "id": id,
            "name": name,
            "created": _now(),
            "status": "queued",
            "error": None,
            "image": image.to_dict(),
            "options": options.to_dict(),
            "stages": {s: StageState().to_dict() for s in STAGES},
            "timings": {"total_s": 0.0},
            "intrinsic_method": None,
            "groups": [],
            "regions_count": 0,
            "palette_id": None,
            "mapping": {},
            "render_options": {},
        }

    @classmethod
    def load(cls, dir: str) -> "Job":
        """Load a job from `<dir>/job.json`; raises OSError/ValueError on a broken record."""
        with open(os.path.join(dir, "job.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict) or "id" not in meta:
            raise ValueError(f"{dir}: job.json is not a job record")
        # Keys added after a job was written get their defaults so the API shape holds.
        template = cls.new_meta(meta["id"], meta.get("name", ""), ImageInfo(0, 0, 0, 0, 0, 0), AnalysisOptions())
        for k, v in template.items():
            meta.setdefault(k, v)
        for s in STAGES:
            meta["stages"].setdefault(s, StageState().to_dict())
        return cls(meta["id"], dir, meta)

    # ------------------------------------------------------------------ paths / io

    def path(self, *parts: str) -> str:
        """Absolute path inside the job directory."""
        return os.path.join(self.dir, *parts)

    @property
    def original_file(self) -> str:
        """Basename of the stored original (`original.jpg` or `original.png`)."""
        return _original_filename(self.meta.get("name", ""))

    def save(self) -> None:
        """Persist `meta` to `job.json` atomically. Safe to call from any number of
        threads at once; a no-op after the job was deleted (the directory is gone and
        must stay gone). The directory itself is created by the registry, never here."""
        with self.lock:
            if self.deleted:
                return
            data = json.dumps(self.meta, indent=1, ensure_ascii=False)
            tmp = self.path(f"job.json.{secrets.token_hex(4)}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(data)
                os.replace(tmp, self.path("job.json"))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

    def snapshot(self) -> dict[str, Any]:
        """A deep copy of `meta`, safe to serialize while the worker keeps mutating."""
        with self.lock:
            return copy.deepcopy(self.meta)

    def summary(self) -> dict[str, Any]:
        """The `JobSummary` row used by `GET /api/jobs`."""
        with self.lock:
            m = self.meta
            return {
                "id": m["id"],
                "name": m["name"],
                "created": m["created"],
                "status": m["status"],
                "thumb": f"/api/jobs/{m['id']}/layers/preview",
                "width": m["image"]["width"],
                "height": m["image"]["height"],
                "n_groups": len(m.get("groups") or []),
            }

    @property
    def options(self) -> AnalysisOptions:
        return AnalysisOptions.from_dict(self.meta.get("options"))

    @property
    def status(self) -> str:
        return self.meta["status"]

    def groups(self) -> list[ColorGroup]:
        """The current color groups as dataclasses (copies; edit through `set_groups`)."""
        with self.lock:
            return [ColorGroup.from_dict(g) for g in self.meta.get("groups") or []]

    # ------------------------------------------------------------------ events

    def subscribe(self) -> queue.Queue:
        """A queue that receives every event published from now on."""
        q: queue.Queue = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        with self.lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def publish(self, event: dict[str, Any]) -> None:
        """Deliver `event` to every subscriber without ever blocking the publisher."""
        with self.lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def replay_events(self) -> list[dict[str, Any]]:
        """The events a late subscriber needs to reconstruct the current state."""
        with self.lock:
            m = self.meta
            out: list[dict[str, Any]] = [{"type": "status", "status": m["status"]}]
            for s in STAGES:
                st = m["stages"][s]
                out.append({"type": "stage", "stage": s, "state": st["state"],
                            "progress": st["progress"], "message": st["message"]})
            if m.get("groups"):
                out.append({"type": "groups", "groups": copy.deepcopy(m["groups"])})
            if m["status"] == "ready":
                out.append({"type": "done"})
            elif m["status"] == "error":
                out.append({"type": "error", "message": m.get("error") or "analysis failed"})
            return out

    # ------------------------------------------------------------------ mutation

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        """Set the job status (and error text), persist, and publish a `status` event
        followed by `done` / `error` when terminal."""
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        with self.lock:
            self.meta["status"] = status
            self.meta["error"] = error
        self.save()
        self.publish({"type": "status", "status": status})
        if status == "ready":
            self.publish({"type": "done"})
        elif status == "error":
            self.publish({"type": "error", "message": error or "analysis failed"})

    def set_stage(self, stage: str, state: str, progress: Optional[float] = None,
                  message: Optional[str] = None) -> None:
        """Update one pipeline stage and publish a `stage` event.

        `running` starts the stage clock (progress defaults to 0); `done`/`error`/
        `skipped` stop it and record `seconds`. Progress ticks while running are
        published but not written to disk (state transitions are), and ticks that
        change nothing visible are coalesced so a chatty model cannot flood clients.
        """
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        if state not in STAGE_STATES:
            raise ValueError(f"bad stage state {state!r}")
        now = time.monotonic()
        with self.lock:
            st = self.meta["stages"][stage]
            transition = st["state"] != state
            if state == "running" and transition:
                self._stage_started[stage] = now
                st["progress"] = 0.0
                st["seconds"] = 0.0
                st["message"] = ""
            if progress is not None:
                st["progress"] = round(float(min(1.0, max(0.0, progress))), 4)
            if message is not None:
                st["message"] = str(message)
            if state == "done":
                st["progress"] = 1.0
            if state in ("done", "error", "skipped"):
                started = self._stage_started.get(stage)
                st["seconds"] = round(now - started, 3) if started is not None else st.get("seconds", 0.0)
                total = sum(float(self.meta["stages"][s].get("seconds") or 0.0) for s in STAGES)
                self.meta["timings"]["total_s"] = round(total, 3)
            st["state"] = state
            if not transition and state == "running":
                last = self._last_tick.get(stage)
                if last is not None and now - last[0] < 0.05 and abs(st["progress"] - last[1]) < 0.01 \
                        and message is None:
                    return
            self._last_tick[stage] = (now, st["progress"])
            event = {"type": "stage", "stage": stage, "state": state,
                     "progress": st["progress"], "message": st["message"]}
        if transition:
            self.save()
        self.publish(event)

    def set_groups(self, groups: list[ColorGroup], regions_count: Optional[int] = None) -> None:
        """Replace the group list, persist, and publish a `groups` event."""
        with self.lock:
            self.meta["groups"] = [g.to_dict() for g in groups]
            if regions_count is not None:
                self.meta["regions_count"] = int(regions_count)
            payload = copy.deepcopy(self.meta["groups"])
        self.save()
        self.publish({"type": "groups", "groups": payload})

    def update(self, **fields: Any) -> None:
        """Set top-level meta keys under the lock and persist."""
        with self.lock:
            self.meta.update(fields)
        self.save()


class JobRegistry:
    """All jobs on disk, loaded lazily from `data/jobs/*/job.json`. Thread-safe.

    `root` defaults to `config.JOBS_DIR` at first use, so tests can point the config
    at a temporary directory before the registry is touched.
    """

    def __init__(self, root: Optional[str] = None):
        self._root = root
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._loaded = False

    @property
    def root(self) -> str:
        if self._root is None:
            self._root = config.JOBS_DIR
        return self._root

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            os.makedirs(self.root, exist_ok=True)
            for entry in sorted(os.listdir(self.root)):
                d = os.path.join(self.root, entry)
                if not os.path.isfile(os.path.join(d, "job.json")):
                    continue
                try:
                    job = Job.load(d)
                except (OSError, ValueError) as e:
                    log.warning("skipping %s: %s", d, e)
                    continue
                self._jobs[job.id] = job

    def create(self, image_rgb_u8: np.ndarray, name: str, options: AnalysisOptions) -> Job:
        """Create a queued job: writes the original (jpg, or png for png uploads) and a
        preview-resolution thumbnail immediately so the gallery can show it before the
        worker gets to it. The array stays on `job.pending_image` for the ingest stage."""
        if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3 or image_rgb_u8.dtype != np.uint8:
            raise ValueError("image must be uint8 HxWx3 RGB")
        self._ensure_loaded()
        h, w = image_rgb_u8.shape[:2]
        ww, wh = imageio.fit_size(w, h, config.WORK_LONG_SIDE)
        pw, ph = imageio.fit_size(w, h, config.PREVIEW_LONG_SIDE)
        info = ImageInfo(width=w, height=h, work_width=ww, work_height=wh, preview_width=pw, preview_height=ph)
        name = os.path.basename(name or "upload.jpg") or "upload.jpg"
        with self._lock:
            jid = new_job_id()
            while jid in self._jobs or os.path.exists(os.path.join(self.root, jid)):
                jid = new_job_id()
            d = os.path.join(self.root, jid)
            os.makedirs(os.path.join(d, "layers"), exist_ok=True)
            os.makedirs(os.path.join(d, "ids"), exist_ok=True)
            os.makedirs(os.path.join(d, "exports"), exist_ok=True)
            job = Job(jid, d, Job.new_meta(jid, name, info, options))
            job.pending_image = image_rgb_u8
            self._jobs[jid] = job
        imageio.save_image(job.path(job.original_file), image_rgb_u8, quality=97)
        imageio.save_image(job.path("preview.jpg"), imageio.resize_to(image_rgb_u8, (pw, ph)), quality=90)
        job.save()
        return job

    def get(self, id: str) -> Optional[Job]:
        """The job with this id, or None (never raises for odd-looking ids)."""
        if not is_job_id(id):
            return None
        self._ensure_loaded()
        with self._lock:
            return self._jobs.get(id)

    def all(self) -> list[Job]:
        """Every job, newest first."""
        self._ensure_loaded()
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.meta.get("created", 0.0), reverse=True)

    def list(self) -> list[dict[str, Any]]:
        """`JobSummary` rows, newest first."""
        return [j.summary() for j in self.all()]

    def delete(self, id: str) -> bool:
        """Remove the job and its directory. Subscribers get a final `status: deleted`
        event so open SSE streams can close. Returns False when there was no such job.

        The job's `deleted` flag is set under its lock *before* the directory is
        removed, so a worker still analyzing it can no longer write `job.json` back
        (which would otherwise resurrect a broken job on the next server start)."""
        self._ensure_loaded()
        with self._lock:
            job = self._jobs.pop(id, None)
        if job is None:
            return False
        with job.lock:
            job.deleted = True
            job.pending_image = None
        job.publish({"type": "status", "status": "deleted"})
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    def __len__(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return len(self._jobs)


registry = JobRegistry()
