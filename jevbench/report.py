from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PRICES, TARGET_ACCURACY

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def load_results(path: str | Path) -> list[dict]:
    """Last record per id (a later successful retry replaces an earlier error)."""
    recs = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r
    return list(recs.values())


def to_frame(records: list[dict], gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in records:
        d = r.get("decision") or {}
        u = r.get("usage") or {}
        rows.append({
            "id": r["id"], "status": r["status"], "model": r.get("model"),
            "pred_queue": d.get("queue"), "pred_priority": d.get("priority"),
            "confidence": d.get("confidence"), "force_human": d.get("force_human", False),
            "flags": ",".join(d.get("flags", []) or []),
            "latency_s": r.get("latency_s"),
            "in_tok": u.get("input_tokens", 0) or 0, "out_tok": u.get("output_tokens", 0) or 0,
            "raw": r.get("raw"), "error": r.get("error"),
        })
    df = gold.merge(pd.DataFrame(rows), on="id", how="left")
    df["valid"] = df["status"].isin(["ok", "recovered"])
    df["queue_ok"] = df["valid"] & (df["pred_queue"] == df["queue"])
    df["priority_ok"] = df["valid"] & (df["pred_priority"] == df["priority"])
    df["both_ok"] = df["queue_ok"] & df["priority_ok"]
    return df


def price_for(model: str | None, system: str) -> dict | None:
    if system == "jev":
        return PRICES["jev"]
    for key, p in PRICES.items():
        if model and key != "jev" and model.startswith(key):
            return p
    return None


def wilson_low(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d


def summarize(df: pd.DataFrame, system: str) -> dict:
    model = str(df["model"].dropna().iloc[0]) if df["model"].notna().any() else "no successful calls"
    price = price_for(model, system)
    if price:
        cost = (df["in_tok"] * price["input"] + df["out_tok"] * price["output"]) / 1e6
        cost_per_1k = float(cost.mean() * 1000)
    else:
        cost_per_1k = float("nan")
    lat = df.loc[df["valid"], "latency_s"].dropna() * 1000 
    return {
        "system": "Jev (typed questions + policy)" if system == "jev" else "LLM (JSON prompt)",
        "model": model,
        "n": len(df),
        "queue accuracy": df["queue_ok"].mean(),
        "priority accuracy": df["priority_ok"].mean(),
        "queue+priority accuracy": df["both_ok"].mean(),
        "invalid outputs %": 100 * (~df["valid"]).mean(),
        "recovered JSON %": 100 * (df["status"] == "recovered").mean(),
        "mean latency ms": lat.mean(),
        "p50 latency ms": lat.median(),
        "p95 latency ms": lat.quantile(0.95),
        "avg input tokens": df["in_tok"].mean(),
        "cost per 1k decisions $": cost_per_1k,
    }


def results_table(summaries: list[dict]) -> pd.DataFrame:
    t = pd.DataFrame(summaries).set_index("system").T
    return t


def format_table(t: pd.DataFrame) -> pd.DataFrame:
    def fmt(idx, v):
        if isinstance(v, str):
            return v
        if "accuracy" in idx:
            return f"{v:.1%}"
        if "%" in idx:
            return f"{v:.1f}%"
        if "ms" in idx or "tokens" in idx:
            return f"{v:,.0f}"
        if "$" in idx:
            return "set price in config.py" if (isinstance(v, float) and math.isnan(v)) else f"${v:,.4f}"
        return str(v)
    return pd.DataFrame({c: [fmt(i, t.loc[i, c]) for i in t.index] for c in t.columns}, index=t.index)


def per_language(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = {}
    for name, df in dfs.items():
        out[name] = df.groupby("lang")["queue_ok"].mean()
    t = pd.DataFrame(out)
    t["n"] = next(iter(dfs.values())).groupby("lang").size()
    return t


def threshold_curve(df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """For each threshold t: auto-decide cases with confidence >= t (and not force_human);
    report accuracy on those and the share they make up of all cases."""
    thresholds = np.round(np.linspace(0, 1, 101), 2) if thresholds is None else thresholds
    conf = df["confidence"].astype(float).fillna(-1)  
    eligible = df["valid"] & ~df["force_human"].fillna(False).astype(bool)
    rows = []
    for t in thresholds:
        auto = eligible & (conf >= t)
        k, n = int(df.loc[auto, "queue_ok"].sum()), int(auto.sum())
        rows.append({"threshold": t, "auto_n": n, "coverage": n / len(df),
                     "accuracy": k / n if n else np.nan, "acc_wilson_low": wilson_low(k, n)})
    return pd.DataFrame(rows)


def operating_point(curve: pd.DataFrame, target: float = TARGET_ACCURACY) -> dict | None:
    """Most automation that stays at >= target accuracy: the lowest threshold t such that the
    auto-decided accuracy is >= target at t AND at every higher threshold (so one lucky
    point on a noisy curve isn't picked)."""
    c = curve[curve["auto_n"] > 0].reset_index(drop=True)
    if c.empty:
        return None
    good = (c["accuracy"] >= target).to_numpy()
    stable = [bool(good[i:].all()) for i in range(len(good))]
    if not any(stable):
        return None
    r = c.iloc[stable.index(True)]
    return {"threshold": float(r.threshold), "auto_n": int(r.auto_n), "coverage": float(r.coverage),
            "accuracy": float(r.accuracy), "acc_wilson_low": float(r.acc_wilson_low)}


def plot_thresholds(curves: dict[str, pd.DataFrame], points: dict[str, dict | None], path: str | Path,
                    title_note: str = "", target: float = TARGET_ACCURACY):
    import matplotlib.pyplot as plt

    colors = [BLUE, ORANGE]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_xlabel("confidence threshold (auto-decide if confidence ≥ threshold)")

    for (name, c), color in zip(curves.items(), colors):
        ax1.plot(c["threshold"], c["accuracy"], color=color, linewidth=2, label=name)
        ax2.plot(c["threshold"], c["coverage"], color=color, linewidth=2, label=name)
        p = points.get(name)
        if p:
            for ax, y in ((ax1, p["accuracy"]), (ax2, p["coverage"])):
                ax.plot([p["threshold"]], [y], "o", color=color, markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            ax2.annotate(f"{p['coverage']:.0%} auto at t={p['threshold']:.2f}",
                         (p["threshold"], p["coverage"]), textcoords="offset points",
                         xytext=(-10, -16), ha="right", color=INK_2, fontsize=9)

    ax1.axhline(target, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(0.01, target + 0.005, f"{target:.0%} target", color=INK_2, fontsize=9, va="bottom")
    ax1.set_ylim(min(0.5, ax1.get_ylim()[0]), 1.01)
    ax1.set_ylabel("accuracy on auto-decided tickets")
    ax1.set_title("Accuracy of what gets automated", loc="left", color=INK, fontsize=11)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("share of all tickets auto-decided")
    ax2.set_title("How much gets automated", loc="left", color=INK, fontsize=11)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 0.93))
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle("Confidence threshold vs accuracy and coverage" + (f"  —  {title_note}" if title_note else ""),
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def tricky_table(jev_df: pd.DataFrame, jev_records: list[dict], llm_df: pd.DataFrame) -> pd.DataFrame:
    by_id = {r["id"]: r for r in jev_records}
    rows = []
    for _, j in jev_df.iterrows():
        l = llm_df.set_index("id").loc[j["id"]]
        ans = (by_id.get(j["id"]) or {}).get("answers") or {}
        topic = ans.get("topic", {})
        top2 = sorted(topic.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:2]
        jev_route = (f"{j['pred_queue']}/{j['pred_priority']} (conf {j['confidence']:.2f})"
                     if j["valid"] else f"ERROR {j['error']}")
        if j["force_human"]:
            jev_route += " → HUMAN (injection flag)"
        rows.append({
            "id": j["id"], "trick": j["trick"], "text": j["text"],
            "expected": f"{j['queue']}/{j['priority']}",
            "Jev decision": jev_route,
            "Jev ok": "✅" if j["both_ok"] else ("◐ queue only" if j["queue_ok"] else "❌"),
            "Jev raw": ("topic " + ", ".join(f"{k} {v:.2f}" for k, v in top2)
                        + f"; unauthorized {ans.get('unauthorized_access', {}).get('noul', float('nan')):.2f}"
                        + f"; urgency {ans.get('urgency', {}).get('score', float('nan')):.2f}"
                        + f"; injection {ans.get('injection_attempt', {}).get('noul', float('nan')):.2f}") if ans else "",
            "LLM decision": (f"{l['pred_queue']}/{l['pred_priority']} (self-conf {l['confidence']})"
                             if l["valid"] else f"INVALID ({l['status']}): {str(l['raw'] or l['error'])[:80]}"),
            "LLM ok": "✅" if l["both_ok"] else ("◐ queue only" if l["queue_ok"] else "❌"),
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Small markdown writer so we don't depend on `tabulate`."""
    d = df.reset_index().rename(columns={"index": "metric"}) if index else df
    cols = [str(c) for c in d.columns]
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)
