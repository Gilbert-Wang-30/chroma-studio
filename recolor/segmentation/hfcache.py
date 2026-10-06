"""Is a pinned Hugging Face snapshot in the local cache? Asked before a model wrapper calls
``from_pretrained``.

``from_pretrained(..., local_files_only=True)`` does not load anything from the network, but
when the files are missing, transformers 5.x still makes an HTTP request while it builds its
error (measured: ``GET https://huggingface.co/api/agent-harnesses``). The wrappers
(:mod:`florence`, :mod:`foreground`, :mod:`matting`, :mod:`partdetect`) look for their weights
again on every job while they are missing, so a fresh install without them would contact the
network at analysis time. :func:`snapshot_present` answers from the cache directory alone
(``huggingface_hub.try_to_load_from_cache``), and a wrapper whose snapshot is missing never calls
``from_pretrained``.
"""
from __future__ import annotations

import os
from typing import Sequence


def snapshot_present(model_id: str, revision: str, files: Sequence[str] = ("config.json",)) -> bool:
    """True when every file of ``files`` of the snapshot ``revision`` (a commit hash) of
    ``model_id`` is in the local Hugging Face cache. Reads the cache directory only; never
    goes to the network. Without ``huggingface_hub`` it returns True, so ``from_pretrained``
    reports the problem the way it always did."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:                  # pragma: no cover - transformers depends on it
        return True
    for name in files:
        try:
            path = try_to_load_from_cache(model_id, name, revision=revision)
        except Exception:  # noqa: BLE001 - a malformed cache entry counts as missing
            return False
        if not isinstance(path, str) or not os.path.isfile(path):
            return False
    return True
