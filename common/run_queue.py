import os
import subprocess
import sys
import time
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPU_MAX, RAM_MAX = 70.0, 70.0
CPU_START, RAM_START = 50.0, 62.0
THREADS = "8"


def log(msg, path=os.path.join(ROOT, "logs", "resources.log")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def read_jobs(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lane, logfile, cmd = [s.strip() for s in line.split("|", 2)]
        jobs.append((lane, logfile, cmd))
    return jobs


def signal_tree(proc, action):
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            getattr(p, action)()
        except psutil.Error:
            pass


def trim(proc):
    import ctypes
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for p in procs:
        h = ctypes.windll.kernel32.OpenProcess(0x0100 | 0x0400, False, p.pid)
        if h:
            ctypes.windll.psapi.EmptyWorkingSet(h)
            ctypes.windll.kernel32.CloseHandle(h)


def other_experiments():
    me = os.getpid()
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if p.info["pid"] != me and "python" in c.lower() and any(
                k in c for k in ("-m Q1.", "-m Q2.", "-m Q3.", "-m Q4.", "common.prepare_data")):
            found.append(p.info["pid"])
    return found


def run(job):
    lane, logfile, cmd = job
    while True:
        cpu, ram = psutil.cpu_percent(1), psutil.virtual_memory().percent
        others = other_experiments()
        if cpu < CPU_START and ram < RAM_START and not others:
            break
        log(f"waiting to start (cpu {cpu:.0f}%, ram {ram:.0f}%, other experiment processes: {len(others)})")
        time.sleep(20)
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS,
               LOKY_MAX_CPU_COUNT=THREADS, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
    os.makedirs(os.path.dirname(os.path.join(ROOT, logfile)), exist_ok=True)
    fh = open(os.path.join(ROOT, logfile), "a")
    p = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc = psutil.Process(p.pid)
    log(f"[{lane}] started (cpu {cpu:.0f}%, ram {ram:.0f}%): {cmd}")
    psutil.cpu_percent(None)
    last = 0
    while p.poll() is None:
        time.sleep(10)
        cpu, ram = psutil.cpu_percent(None), psutil.virtual_memory().percent
        if cpu > CPU_MAX:
            signal_tree(proc, "suspend")
            log(f"CPU {cpu:.0f}% > {CPU_MAX}%: job paused for 30 s")
            time.sleep(30)
            signal_tree(proc, "resume")
        if ram > 65.0:
            trim(proc)
            ram_after = psutil.virtual_memory().percent
            if ram_after > RAM_MAX:
                log(f"WARNING RAM {ram_after:.1f}% > {RAM_MAX}% even after trimming")
            ram = ram_after
        if time.time() - last > 30:
            log(f"cpu {cpu:5.1f}%  ram {ram:5.1f}%  running: {cmd[:70]}")
            last = time.time()
    fh.close()
    log(f"[{lane}] finished rc={p.returncode}: {cmd}")
    return p.returncode


def main(jobs_path, start_at=0):
    i = start_at
    while True:
        jobs = read_jobs(jobs_path)
        if i >= len(jobs):
            break
        run(jobs[i])
        i += 1
    log("queue empty")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)

