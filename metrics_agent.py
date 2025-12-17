#!/usr/bin/env python3
"""
metrics_agent.py
Runs on each VM and exposes:
  POST /start
  POST /stop
  GET  /dump?clear=1&gpu=0

Collects:
- CPU %
- RAM usage
- Network RX/TX (bytes/sec)
- Optional ping RTT to targets provided via PEER_TARGETS env var

Automation-safe: NO interactive input().
"""

import os
import time
import psutil
import platform
import threading
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

STATE = {
    "running": False,
    "thread": None,
    "lock": threading.Lock(),
    "samples": [],
    "t0": None,
}

# Optional: RTT targets for ping
# export PEER_TARGETS="10.0.0.14,10.0.0.5,10.0.0.6"
PEER_TARGETS = [x.strip() for x in os.environ.get("PEER_TARGETS", "").split(",") if x.strip()]
SAMPLE_INTERVAL_S = float(os.environ.get("SAMPLE_INTERVAL_S", "1.0"))

PING_COUNT = int(os.environ.get("PING_COUNT", "1"))
PING_TIMEOUT_S = float(os.environ.get("PING_TIMEOUT_S", "1.0"))


def _read_net_bytes():
    c = psutil.net_io_counters()
    return {"bytes_sent": c.bytes_sent, "bytes_recv": c.bytes_recv}


def _ping_once(host: str):
    """Return avg RTT ms (float) or None."""
    try:
        cmd = ["ping", "-n", "-c", str(PING_COUNT), "-W", str(int(max(1, PING_TIMEOUT_S))), host]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=PING_TIMEOUT_S + 1).decode("utf-8", "ignore")

        for ln in out.splitlines():
            if "rtt min/avg/max" in ln:
                stats = ln.split("=")[-1].strip().split()[0]
                avg = float(stats.split("/")[1])
                return avg

        for ln in out.splitlines():
            if "time=" in ln:
                part = ln.split("time=")[-1].split()[0]
                return float(part)

    except Exception:
        return None

    return None


def _collector():
    STATE["t0"] = time.time()
    last_net = _read_net_bytes()
    last_t = time.time()

    psutil.cpu_percent(interval=None)

    while True:
        with STATE["lock"]:
            if not STATE["running"]:
                break

        now = time.time()
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()

        net = _read_net_bytes()
        dt = max(1e-6, now - last_t)
        tx_Bps = (net["bytes_sent"] - last_net["bytes_sent"]) / dt
        rx_Bps = (net["bytes_recv"] - last_net["bytes_recv"]) / dt
        last_net, last_t = net, now

        rtts = {}
        for host in PEER_TARGETS:
            rtts[host] = _ping_once(host)

        sample = {
            "ts": now,
            "uptime_s": round(now - STATE["t0"], 3),
            "cpu_percent": cpu,
            "ram_used_bytes": vm.used,
            "ram_available_bytes": vm.available,
            "ram_percent": vm.percent,
            "net_tx_Bps": tx_Bps,
            "net_rx_Bps": rx_Bps,
            "rtt_ms": rtts,
            "host": platform.node(),
        }

        with STATE["lock"]:
            STATE["samples"].append(sample)

        time.sleep(SAMPLE_INTERVAL_S)


@app.route("/start", methods=["POST"])
def start():
    with STATE["lock"]:
        if STATE["running"]:
            return jsonify({"ok": True, "msg": "already running"})
        STATE["running"] = True
        STATE["samples"].clear()
        th = threading.Thread(target=_collector, daemon=True)
        STATE["thread"] = th
        th.start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    with STATE["lock"]:
        STATE["running"] = False
    return jsonify({"ok": True})


@app.route("/dump", methods=["GET"])
def dump():
    clear = request.args.get("clear", "0") == "1"
    gpu = request.args.get("gpu", "0") == "1"  # compatibility; ignored

    with STATE["lock"]:
        data = {
            "meta": {
                "host": platform.node(),
                "peer_targets": PEER_TARGETS,
                "sample_interval_s": SAMPLE_INTERVAL_S,
                "gpu_enabled": gpu,
                "count": len(STATE["samples"]),
            },
            "samples": list(STATE["samples"]),
        }
        if clear:
            STATE["samples"].clear()
    return jsonify(data)


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", "5001"))
    app.run(host="0.0.0.0", port=port)
