#!/usr/bin/env python3
from flask import Flask, jsonify, request
import psutil, socket, time, threading, os
import socket as pysocket

# --- Optional GPU (NVML) support ---
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_OK = True
except Exception:
    GPU_OK = False

app = Flask(__name__)

# --- Monitoring state ---
collecting = False
buffer = []
interval = 0.5
thread = None

_last_net = psutil.net_io_counters()
_last_time = time.time()

def _tcp_latency_ms(target: str, port: int, timeout=1.0):
    """Measure TCP connect latency."""
    if not target:
        return None
    s = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_STREAM)
    s.settimeout(timeout)
    start = time.monotonic()
    try:
        s.connect((target, int(port)))
        return (time.monotonic() - start) * 1000.0
    except Exception:
        return None
    finally:
        try:
            s.close()
        except Exception:
            pass

def _petals_processes():
    """Find Petals processes by cmdline."""
    procs = []
    for p in psutil.process_iter(attrs=["name", "cmdline", "pid"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            name = (p.info.get("name") or "").lower()
            if "petals" in cmd and ("run_server" in cmd or "petals.cli.run_server" in cmd):
                procs.append(p)
            elif "petals" in name:
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs

def _snapshot():
    """Take one snapshot of all metrics."""
    global _last_net, _last_time

    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    net = psutil.net_io_counters()

    # --- Flow deltas ---
    now = time.time()
    elapsed = max(1e-6, now - _last_time)
    bytes_sent_delta = max(0, net.bytes_sent - _last_net.bytes_sent)
    bytes_recv_delta = max(0, net.bytes_recv - _last_net.bytes_recv)
    pkts_sent_delta = max(0, net.packets_sent - _last_net.packets_sent)
    pkts_recv_delta = max(0, net.packets_recv - _last_net.packets_recv)

    flow_sent_mb = bytes_sent_delta / (1024 * 1024)
    flow_recv_mb = bytes_recv_delta / (1024 * 1024)
    flow_throughput_mbps = ((bytes_sent_delta + bytes_recv_delta) * 8) / (1_000_000 * elapsed)

    _last_net, _last_time = net, now

    # --- Petals metrics ---
    petals_cpu, petals_mem_mb, petals_threads = 0.0, 0.0, 0
    petals_proc_count = 0
    procs = _petals_processes()
    petals_proc_count = len(procs)
    for p in procs:
        try:
            petals_cpu += p.cpu_percent(None)
            petals_mem_mb += p.memory_info().rss / (1024 * 1024)
            petals_threads += p.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # --- GPU metrics ---
    gpus = []
    if GPU_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # milliwatts → watts
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                bw_util = util.memory  # GPU memory bandwidth utilization (%)
                pci_tx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0  # KB/s → MB/s
                pci_rx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0  # KB/s → MB/s

                gpus.append({
                    "index": i,
                    "util_percent": util.gpu,
                    "mem_used_mb": int(mem.used / (1024 * 1024)),
                    "mem_total_mb": int(mem.total / (1024 * 1024)),
                    "mem_bandwidth_util": bw_util,
                    "temperature_C": temp,
                    "power_W": round(power, 2),
                    "power_limit_W": round(power_limit, 2),
                    "pcie_tx_MBps": round(pci_tx, 2),
                    "pcie_rx_MBps": round(pci_rx, 2),
                })
        except Exception as e:
            print(f"GPU metrics collection failed: {e}")
            gpus = []

        except Exception:
            gpus = []

    return {
        "timestamp": now,
        "hostname": socket.gethostname(),
        "cpu_percent": cpu_percent,
        "memory_used_mb": int(mem.used / (1024*1024)),
        "memory_percent": mem.percent,
        "bytes_sent": net.bytes_sent,
        "bytes_recv": net.bytes_recv,
        "packets_sent": net.packets_sent,
        "packets_recv": net.packets_recv,
        "flow_sent_mb": round(flow_sent_mb,3),
        "flow_recv_mb": round(flow_recv_mb,3),
        "flow_throughput_mbps": round(flow_throughput_mbps,3),
        "bytes_sent_delta": bytes_sent_delta,
        "bytes_recv_delta": bytes_recv_delta,
        "packets_sent_delta": pkts_sent_delta,
        "packets_recv_delta": pkts_recv_delta,
        "petals_proc_count": petals_proc_count,
        "petals_cpu_percent": petals_cpu,
        "petals_mem_mb": int(petals_mem_mb),
        "petals_threads": petals_threads,
        "gpus": gpus
    }

def _collector_loop():
    global collecting, buffer
    while collecting:
        buffer.append(_snapshot())
        time.sleep(interval)

@app.route("/start", methods=["POST"])
def start():
    global collecting, buffer, thread
    buffer = []
    collecting = True
    thread = threading.Thread(target=_collector_loop, daemon=True)
    thread.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global collecting
    collecting = False
    return jsonify({"status": "stopped"})

@app.route("/dump", methods=["GET"])
def dump():
    metrics = {
        "hostname": [],
        "cpu_percent": [],
        "memory_used_mb": [],
        "memory_percent": [],
        "bytes_sent": [],
        "bytes_recv": [],
        "packets_sent": [],
        "packets_recv": [],
        "flow_sent_mb": [],
        "flow_recv_mb": [],
        "flow_throughput_mbps": [],
        "bytes_sent_delta": [],
        "bytes_recv_delta": [],
        "packets_sent_delta": [],
        "packets_recv_delta": [],
        "petals_proc_count": [],
        "petals_cpu_percent": [],
        "petals_mem_mb": [],
        "petals_threads": [],
        "gpus": []
    }

    for i in buffer:
        for j in metrics:
            metrics[j].append(i[j])


    for i in metrics:
        if i not in ["hostname", "gpus"] :
            metrics[i] = sum(metrics[i])/len(metrics[i])
        elif i=="hostname":
            metrics["hostname"] = socket.gethostname()

    return jsonify(metrics)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
