#!/usr/bin/env python3
from flask import Flask, jsonify, request
import psutil
import socket
import time
import threading
import os
import socket as pysocket

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_OK = True
except:
    GPU_OK = False

app = Flask(__name__)

INTERVAL = float(os.environ.get("INTERVAL", "1.0"))

AGENTS = {
    "VM0": "http://10.0.0.4:5001",
    "VM1": "http://10.0.0.14:5001",
    "VM2": "http://10.0.0.5:5001",
    "VM3": "http://10.0.0.6:5001",
    "VM4": "http://10.0.0.7:5001",
    "VM5": "http://10.0.0.8:5001",
    "VM6": "http://10.0.0.9:5001",
    "VM7": "http://10.0.0.10:5001",
    "VM8": "http://10.0.0.11:5001",
    "VM9": "http://10.0.0.12:5001",
    "VM10": "http://10.0.0.13:5001",
}

x = ",".join([AGENTS[i].replace("http://", "") for i in ["VM1", "VM2", "VM4"]])

# Example: export PEER_TARGETS="10.0.0.5:5001,10.0.0.6:5001"
_PEER_TARGETS = os.environ.get("PEER_TARGETS", x)
PEER_TARGETS = [t.strip() for t in _PEER_TARGETS.split(",") if t.strip()]

collecting = False
thread = None

_last_net = psutil.net_io_counters()
_last_time = time.time()


class Stat:
    __slots__=["n","mean","min","max"]
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.min = float("inf")
        self.max = float("-inf")
    def add(self, x):
        self.n += 1
        self.mean += (x - self.mean)/self.n
        self.min = min(self.min, x)
        self.max = max(self.max, x)
    def as_dict(self):
        if self.n == 0:
            return {"count":0,"avg":0,"min":0,"max":0}
        return {
            "count": self.n,
            "avg": round(self.mean,3),
            "min": self.min,
            "max": self.max
        }


stats = {}
gpu_stats = {}

def _S(key):
    if key not in stats:
        stats[key] = Stat()
    return stats[key]

def _SG(key, idx):
    d = gpu_stats.setdefault(key, {})
    if idx not in d:
        d[idx] = Stat()
    return d[idx]

def _tcp_latency_ms(host, port, timeout=0.5):
    s = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_STREAM)
    s.settimeout(timeout)
    t0 = time.monotonic()
    try:
        s.connect((host, int(port)))
        return (time.monotonic() - t0)*1000
    except:
        return None
    finally:
        try: s.close()
        except: pass


def _snapshot():
    global _last_net, _last_time

    cpu_percent = psutil.cpu_percent(None)
    mem = psutil.virtual_memory()

    net = psutil.net_io_counters()
    now = time.time()
    dt = max(1e-6, now - _last_time)

    sent_delta = max(0, net.bytes_sent - _last_net.bytes_sent)
    recv_delta = max(0, net.bytes_recv - _last_net.bytes_recv)

    flow_sent_mb = sent_delta/(1024*1024)
    flow_recv_mb = recv_delta/(1024*1024)
    flow_throughput_mbps = ((sent_delta+recv_delta)*8)/(1_000_000*dt)

    _last_net, _last_time = net, now

    # Petals process
    petals_cpu = petals_mem_mb = petals_threads = petals_proc_count = 0
    for p in psutil.process_iter(attrs=["name","cmdline","pid"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "petals" in cmd and ("run_server" in cmd or "petals.cli.run_server" in cmd):
                petals_proc_count += 1
                petals_cpu += p.cpu_percent(None)
                petals_mem_mb += p.memory_info().rss/(1024*1024)
                petals_threads += p.num_threads()
        except:
            continue

    _S("cpu_percent").add(cpu_percent)
    _S("memory_used_mb").add(int(mem.used/(1024*1024)))
    _S("memory_percent").add(mem.percent)
    _S("flow_sent_mb").add(flow_sent_mb)
    _S("flow_recv_mb").add(flow_recv_mb)
    _S("flow_throughput_mbps").add(flow_throughput_mbps)
    _S("petals_proc_count").add(petals_proc_count)
    _S("petals_cpu_percent").add(petals_cpu)
    _S("petals_mem_mb").add(int(petals_mem_mb))
    _S("petals_threads").add(petals_threads)

    # RTT per peer
    for entry in PEER_TARGETS:
        try:
            host,port = entry.split(":")
        except:
            continue
        rtt = _tcp_latency_ms(host, port)
        if rtt is not None:
            key = f"peer_rtt_ms_{host.replace('.', '_')}_{port}"
            _S(key).add(rtt)

    # GPU
    if GPU_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                memi = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                _SG("gpu_util_percent", i).add(util.gpu)
                _SG("gpu_mem_used_mb", i).add(int(memi.used/(1024*1024)))
                _SG("gpu_mem_total_mb").add(int(memi.total/(1024*1024)))
        except:
            pass


def _collector_loop():
    global collecting
    while collecting:
        _snapshot()
        time.sleep(INTERVAL)


@app.route("/start", methods=["POST"])
def start():
    global collecting, thread, stats, gpu_stats
    stats.clear()
    gpu_stats.clear()
    collecting = True
    thread = threading.Thread(target=_collector_loop, daemon=True)
    thread.start()
    return jsonify({"status":"started"})


@app.route("/stop", methods=["POST"])
def stop():
    global collecting
    collecting = False
    return jsonify({"status":"stopped"})


@app.route("/dump", methods=["GET"])
def dump():
    include_gpu = request.args.get("gpu","1")!="0"
    clear = request.args.get("clear","1")!="0"

    out = {
        "hostname": socket.gethostname(),
        "metrics": {k:v.as_dict() for k,v in stats.items()}
    }

    if include_gpu:
        out["gpu_metrics"] = {
            k:{idx:st.as_dict() for idx,st in d.items()}
            for k,d in gpu_stats.items()
        }

    if clear:
        stats.clear()
        gpu_stats.clear()

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
