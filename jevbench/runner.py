from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .llm import parse_llm_output
from .policy import decide


def _load_done(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if not rec.get("error"):
                done[rec["id"]] = rec
    return done


def run_system(system, df: pd.DataFrame, out_path: str | Path, warmup: bool = True, verbose: bool = True) -> list[dict]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path)
    todo = df[~df["id"].isin(done)]

    if warmup and len(todo):
        try:
            system.ask("Hello, where is my order?")
        except Exception as e:
            print(f"[warmup] {type(e).__name__}: {e}")

    pause = float(getattr(system, "min_interval", 0) or 0)   
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            if pause and i > 1:
                time.sleep(pause)
            rec = {"id": row.id, "system": system.name}
            t0 = time.perf_counter()
            try:
                result, usage, model = system.ask(row.text)
                rec["latency_s"] = time.perf_counter() - t0
                rec["usage"], rec["model"] = usage, model
                if system.name == "jev":
                    rec["answers"] = result
                    rec["decision"] = decide(result)
                    rec["status"] = "ok"
                else:
                    rec["raw"] = result
                    parsed, status = parse_llm_output(result)
                    rec["decision"], rec["status"] = parsed, status
            except Exception as e:
                rec["latency_s"] = time.perf_counter() - t0
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
                rec["status"] = "api_error"
                rec["decision"] = None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[row.id] = rec
            if i == 3 and all(done[r].get("status") == "api_error" for r in todo["id"].iloc[:3]):
                raise RuntimeError(f"First 3 calls to {system.name} failed - check your key/backend. "
                                   f"Last error: {rec['error']}")
            if verbose:
                d = rec.get("decision") or {}
                print(f"{system.name} {i:>3}/{len(todo)} {row.id} {rec['status']:<16} "
                      f"{d.get('queue', '-'):<10} {rec['latency_s']*1000:7.0f} ms")
    return [done[i] for i in df["id"] if i in done]
