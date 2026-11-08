from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .queue import TASKS, connect, enqueue, state_dir


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    # Configuration saved in the console must be read afresh by each job.
    path = Path(env.get("KB_ENV_FILE", "/data/config/knowledge-base.env"))
    for line in path.read_text().splitlines() if path.exists() else []:
        key, sep, _ = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            env.pop(key.strip(), None)
    for key in (
        "KB_LOCAL_BASE_DIR",
        "KB_ENV_FILE",
        "KB_RUNTIME",
        "KB_RUNTIME_BOOT_TIME",
        "KB_APP_DIR",
        "KB_PYTHON",
        "KB_WEB_HOST",
        "KB_SEARCH_HOST",
        "KB_STORAGE_BACKEND",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def command(kind, payload):
    if kind in TASKS:
        return [
            "/bin/bash",
            str(
                Path(os.getenv("KB_SCRIPTS_DIR", "/opt/carrel/scripts"))
                / TASKS[kind][2][0]
            ),
            *TASKS[kind][2][1:],
        ]
    cmd = [
        sys.executable,
        "-m",
        "kb_pipeline",
        "--env-file",
        os.environ["KB_ENV_FILE"],
        "graph",
        "append" if payload.get("append") else "build",
        "--source",
        payload["kb_id"],
    ]
    if payload.get("graph_version") and not payload.get("append"):
        cmd += [
            "--graph-version",
            payload["graph_version"],
            "--allow-existing-graph-version",
        ]
    if payload.get("force") and payload.get("append"):
        cmd += ["--force"]
    return cmd


def group(kind):
    return (
        "graph"
        if kind in ("graph", "graph-build")
        else ("maintenance" if kind in ("gc", "cache", "logs") else kind)
    )


def recover_requests():
    # Periodic rounds are idempotent. Graph recovery uses the pipeline's existing
    # version/cache recovery, including its distinction between pause and shutdown.
    with connect() as con:
        con.execute(
            "UPDATE requests SET status='interrupted',finished=? WHERE status='running'",
            (time.time(),),
        )
    enqueue("scan")
    enqueue("worker")
    enqueue("graph")


class Runtime:
    def __init__(self):
        self.stopping = False
        self.children = {}
        self.retry_at = {}
        self.log_dir = Path(
            os.getenv("KB_LOG_DIR", str(Path(os.environ["KB_LOCAL_BASE_DIR"]) / "logs"))
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def spawn(self, key, cmd, request=None):
        log = self.log_dir / (
            "runtime-" + (request["kind"] if request else key) + ".log"
        )
        with log.open("ab") as output:
            proc = subprocess.Popen(
                cmd,
                env=child_env(),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children[key] = (proc, request)
        return proc

    def tick(self, now):
        for key, (proc, request) in list(self.children.items()):
            code = proc.poll()
            if code is None:
                if (
                    request
                    and request["kind"] == "graph-build"
                    and not request.get("timed_out")
                    and now - request["started"]
                    >= float(os.getenv("KB_GRAPH_BUILD_MAX_HOURS", "48")) * 3600
                ):
                    os.killpg(proc.pid, signal.SIGTERM)
                    request["timed_out"] = True
                continue
            del self.children[key]
            if request:
                with connect() as con:
                    con.execute(
                        "UPDATE requests SET status=?,finished=?,result=? WHERE id=? AND status='running'",
                        ("done" if code == 0 else "failed", now, code, request["id"]),
                    )
                    con.execute(
                        "UPDATE schedules SET last_finished=?,result=? WHERE kind=?",
                        (now, code, request["kind"]),
                    )
            else:
                self.retry_at[key] = now + 5
                print(f"[runtime] {key} exited {code}; restarting in 5s", flush=True)
        for key, module in (("web", os.getenv("KB_WEB_MODULE", "kb_server")), ("search", "kb_search")):
            if key not in self.children and now >= self.retry_at.get(key, 0):
                self.spawn(key, [sys.executable, "-m", module])
        with connect() as con:
            for kind, (interval, delay, _) in TASKS.items():
                con.execute(
                    "INSERT OR IGNORE INTO schedules(kind,next_due) VALUES(?,?)",
                    (kind, now + delay),
                )
            due = list(
                con.execute("SELECT kind FROM schedules WHERE next_due<=?", (now,))
            )
        for row in due:
            kind = row["kind"]
            with connect() as con:
                busy = con.execute(
                    "SELECT 1 FROM requests WHERE kind=? AND status IN ('queued','running')",
                    (kind,),
                ).fetchone()
            if not busy:
                enqueue(kind)
            with connect() as con:
                con.execute(
                    "UPDATE schedules SET next_due=? WHERE kind=?",
                    (now + TASKS[kind][0], kind),
                )
        occupied = {group(r["kind"]) for _, r in self.children.values() if r}
        with connect() as con:
            pending = list(
                con.execute("SELECT * FROM requests WHERE status='queued' ORDER BY id")
            )
        for row in pending:
            kind = row["kind"]
            family = group(kind)
            if family in occupied:
                continue
            # Maintenance and graph work share the existing locks/yield rules.
            if family == "maintenance" and occupied & {"worker", "graph", "scan"}:
                continue
            if family in ("worker", "graph", "scan") and "maintenance" in occupied:
                continue
            request = dict(row)
            request["started"] = now
            payload = json.loads(row["payload"])
            with connect() as con:
                # A pause may cancel a request after the pending list was read.
                con.execute("BEGIN IMMEDIATE")
                if (
                    con.execute(
                        "SELECT status FROM requests WHERE id=?", (row["id"],)
                    ).fetchone()[0]
                    != "queued"
                ):
                    continue
                try:
                    proc = self.spawn(
                        "job:" + str(row["id"]), command(kind, payload), request
                    )
                except OSError as exc:
                    print(f"[runtime] Cannot start {kind}: {exc}", flush=True)
                    con.execute(
                        "UPDATE requests SET status='failed',finished=?,result=-1 WHERE id=?",
                        (now, row["id"]),
                    )
                    continue
                con.execute(
                    "UPDATE requests SET status='running',pid=?,started=? WHERE id=?",
                    (proc.pid, now, row["id"]),
                )
                con.execute(
                    "UPDATE schedules SET last_started=? WHERE kind=?", (now, kind)
                )
            occupied.add(family)
        with connect() as con:
            con.execute(
                "DELETE FROM requests WHERE finished<? AND status NOT IN ('queued','running')",
                (now - 30 * 86400,),
            )
        self.heartbeat(now)

    def heartbeat(self, now):
        heartbeat = {
            "at": now,
            "stopping": self.stopping,
            "services": {k: p.pid for k, (p, r) in self.children.items() if not r},
        }
        path = state_dir() / "runtime.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(heartbeat))
        temp.replace(path)

    def stop(self):
        self.stopping = True
        self.heartbeat(time.time())
        (state_dir() / "container-stopping").touch()
        for proc, _ in self.children.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + float(
            os.getenv("KB_SHUTDOWN_GRACE_SECONDS", "45")
        )

        def groups_alive():
            alive = False
            for p, _ in self.children.values():
                p.poll()  # Reap parents; shell grandchildren may still be flushing checkpoints.
                try:
                    os.killpg(p.pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            return alive

        while groups_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        for proc, _ in self.children.values():
            # Reap entire groups, including children of the shell launchers.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "request" and len(sys.argv) == 3:
            print(enqueue(sys.argv[2]))
            return
        if sys.argv[1] == "health":
            from .bootstrap import health

            health()
            return
        raise SystemExit("usage: python -m kb_runtime [health | request TASK]")
    from .bootstrap import initialize

    state_dir().mkdir(parents=True, exist_ok=True)
    lock = (state_dir() / "runtime.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another Carrel runtime owns this data directory")
    initialize()
    os.environ["KB_RUNTIME_BOOT_TIME"] = str(int(time.time()))
    (state_dir() / "container-stopping").unlink(missing_ok=True)
    runtime = Runtime()

    def stop_signal(*_):
        runtime.stopping = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    recover_requests()
    try:
        while not runtime.stopping:
            runtime.tick(time.time())
            time.sleep(0.5)
    finally:
        runtime.stop()
        lock.close()


if __name__ == "__main__":
    main()
