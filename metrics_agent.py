#!/usr/bin/env python3
from flask import Flask, jsonify, request
import psutil, socket, time, threading, os
import socket as pysocket

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_OK = True
except Exception:
    GPU_OK = False

app = Flask(__name__)

# ---- Config ----
INTERVAL = float(os.environ.get("INTERVAL", "1.0"))  # seconds
# ----------------

collecting = False
thread = None
_last_net = psutil.net_io_counters()
_last_time = time.time()


class Stat:
    __slots__ = ("n", "mean", "min", "max")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def add(self, x: float):
        self.n += 1
        # Welford mean
        self.mean += (x - self.mean) / self.n
        if x < self.min: self.min = x
        if x > self.max: self.max = x

    def as_dict(self):
        if self.n == 0:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}
        return {"count": self.n, "avg": round(self.mean, 3), "min": self.min, "max": self.max}


stats = {}  # key -> Stat
gpu_stats = {}  # key -> {gpu_index: Stat}


def _S(key):
    if key not in stats: stats[key] = Stat()
    return stats[key]


def _SG(key, idx):
    d = gpu_stats.setdefault(key, {})
    if idx not in d: d[idx] = Stat()
    return d[idx]


def _tcp_latency_ms(target: str, port: int, timeout=0.5):
    if not target: return None
    s = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_STREAM)
    s.settimeout(timeout)
    t0 = time.monotonic()
    try:
        s.connect((target, int(port)))
        return (time.monotonic() - t0) * 1000.0
    except Exception:
        return None
    finally:
        try:
            s.close()
        except Exception:
            pass


def _snapshot():
    global _last_net, _last_time
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    net = psutil.net_io_counters()

    now = time.time()
    elapsed = max(1e-6, now - _last_time)
    bytes_sent_delta = max(0, net.bytes_sent - _last_net.bytes_sent)
    bytes_recv_delta = max(0, net.bytes_recv - _last_net.bytes_recv)
    flow_sent_mb = bytes_sent_delta / (1024 * 1024)
    flow_recv_mb = bytes_recv_delta / (1024 * 1024)
    flow_throughput_mbps = ((bytes_sent_delta + bytes_recv_delta) * 8) / (1_000_000 * elapsed)

    _last_net, _last_time = net, now

    # Petals process summary
    petals_cpu, petals_mem_mb, petals_threads = 0.0, 0.0, 0
    petals_proc_count = 0
    for p in psutil.process_iter(attrs=["name", "cmdline", "pid"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            name = (p.info.get("name") or "").lower()
            if "petals" in cmd and ("run_server" in cmd or "petals.cli.run_server" in cmd):
                petals_proc_count += 1
                petals_cpu += p.cpu_percent(None)
                petals_mem_mb += p.memory_info().rss / (1024 * 1024)
                petals_threads += p.num_threads()
            elif "petals" in name:
                petals_proc_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Update scalar stats
    _S("cpu_percent").add(cpu_percent)
    _S("memory_used_mb").add(int(mem.used / (1024 * 1024)))
    _S("memory_percent").add(mem.percent)
    _S("flow_sent_mb").add(flow_sent_mb)
    _S("flow_recv_mb").add(flow_recv_mb)
    _S("flow_throughput_mbps").add(flow_throughput_mbps)
    _S("petals_proc_count").add(petals_proc_count)
    _S("petals_cpu_percent").add(petals_cpu)
    _S("petals_mem_mb").add(int(petals_mem_mb))
    _S("petals_threads").add(petals_threads)

    # GPU stats (averaged per GPU index)
    if GPU_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                memi = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                _SG("gpu_util_percent", i).add(util.gpu)
                _SG("gpu_mem_used_mb", i).add(int(memi.used / (1024 * 1024)))
                _SG("gpu_mem_total_mb", i).add(int(memi.total / (1024 * 1024)))
        except Exception:
            pass


def _collector_loop():
    global collecting
    while collecting:
        _snapshot()
        time.sleep(INTERVAL)


@app.route("/start", methods=["POST"])
def start():
    global collecting, thread, stats, gpu_stats
    stats = {}
    gpu_stats = {}
    collecting = True
    thread = threading.Thread(target=_collector_loop, daemon=True)
    thread.start()
    return jsonify({"status": "started", "interval": INTERVAL})


@app.route("/stop", methods=["POST"])
def stop():
    global collecting
    collecting = False
    return jsonify({"status": "stopped"})


@app.route("/dump", methods=["GET"])
def dump():
    """Return aggregates and (by default) clear them."""
    include_gpu = request.args.get("gpu", "1") != "0"
    clear = request.args.get("clear", "1") != "0"

    out = {
        "hostname": socket.gethostname(),
        "metrics": {k: v.as_dict() for k, v in stats.items()}
    }
    if include_gpu and gpu_stats:
        out["gpu_metrics"] = {
            k: {idx: st.as_dict() for idx, st in d.items()}
            for k, d in gpu_stats.items()
        }
    if clear:
        stats.clear()
        gpu_stats.clear()
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
