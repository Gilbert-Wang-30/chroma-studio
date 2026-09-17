#!/usr/bin/env python
"""Run the Recolor server on every interface and print where to open it.

    .venv/bin/python serve.py [--port N] [--warmup]

Binds 0.0.0.0:config.SERVER_PORT (env RECOLOR_PORT, default 8810) and prints the
loopback, LAN and Tailscale URLs. SAM 2 and the intrinsic model load lazily on first
use (the first analysis, or a full-resolution export) and are dropped again after
config.IDLE_UNLOAD_S seconds (env RECOLOR_IDLE_UNLOAD_S, default 120) with none
running, so the GPU only holds them while the app is actually being used. Pass
--warmup to preload both at startup instead of waiting for the first job; they are
still unloaded on the same idle timer afterwards. Reminds you about ufw when the LAN
port is closed.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from recolor import config  # noqa: E402


def lan_address() -> str | None:
    """The address a LAN peer would use: the source address of a UDP socket pointed
    at a public IP (no packet is sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def tailscale_address() -> str | None:
    """The machine's Tailscale IPv4, or None when tailscale is absent or down."""
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, timeout=3, text=True)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return lines[0] if out.returncode == 0 and lines else None
    except (OSError, subprocess.SubprocessError):
        return None


def local_addresses() -> list[tuple[str, str]]:
    """`[(address, label)]` in the order worth printing; loopback first, no duplicates,
    Docker-style 172.16-31.x addresses skipped."""
    found: list[tuple[str, str]] = [("127.0.0.1", "this machine")]
    lan = lan_address()
    if lan:
        found.append((lan, "same network"))
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append((info[4][0], "same network"))
    except OSError:
        pass
    ts = tailscale_address()
    if ts:
        found.append((ts, "over Tailscale, from any of your devices"))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for addr, label in found:
        if addr in seen or addr.startswith("172.1") or addr.startswith("172.2") or addr.startswith("172.3"):
            continue
        if addr.startswith("100.") and label == "same network":
            label = "over Tailscale, from any of your devices"
        seen.add(addr)
        out.append((addr, label))
    return out


def firewall_is_up() -> bool:
    """True when ufw is enabled (its config says so); we cannot tell which ports are open
    without root, so the hint is printed whenever it is on."""
    try:
        with open("/etc/ufw/ufw.conf", "r", encoding="utf-8") as f:
            return "ENABLED=yes" in f.read()
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Recolor server")
    ap.add_argument("--port", type=int, default=config.SERVER_PORT)
    ap.add_argument("--warmup", action="store_true",
                    help="preload SAM 2 / Intrinsic at startup instead of on the first job "
                         "(they are still unloaded after the idle timeout)")
    ap.add_argument("--no-warmup", action="store_true", help=argparse.SUPPRESS)  # deprecated: already the default
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    import logging
    import uvicorn
    from recolor.server.app import create_app

    # uvicorn configures only its own loggers; the pipeline's stage warnings, fallbacks
    # and "deleted mid-analysis" notes live under `recolor.*` and would otherwise vanish.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s:     [%(name)s] %(message)s"))
    rlog = logging.getLogger("recolor")
    rlog.addHandler(handler)
    rlog.propagate = False   # some model libraries call basicConfig(); avoid double lines
    rlog.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))

    app = create_app(warmup=args.warmup)
    addresses = local_addresses()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # the banner must show up even when piped to a log
    except (AttributeError, ValueError):
        pass

    print()
    print("  Recolor - Chroma Studio")
    for addr, label in addresses:
        print(f"    http://{addr}:{args.port}    {label}")
    print(f"    http://{addresses[-1][0]}:{args.port}/api/docs    API reference")
    print()
    idle_s = config.IDLE_UNLOAD_S
    idle_note = f"idle {idle_s:g}s" if idle_s > 0 else "idle-unload disabled"
    if args.warmup:
        print(f"  Warming SAM 2 and Intrinsic on the GPU now; unloaded again after {idle_note} unused.")
    else:
        print(f"  Models load on the first job and unload again after {idle_note} unused.")
    if firewall_is_up():
        print()
        print(f"  ufw is active: if the LAN address does not answer, open the port with")
        print(f"      sudo ufw allow {args.port}/tcp")
        print("  Tailscale addresses work regardless.")
    print()
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.log_level, access_log=False)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        print("stopped")


if __name__ == "__main__":
    main()
