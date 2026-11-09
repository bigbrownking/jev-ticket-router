"""Run the whole assignment: Jev and an LLM on the labeled set and the tricky set, then write
the results table, the confidence-threshold plot and the tricky-case table to results/.

Examples
  # Jev through TypeSafe's API + Claude Sonnet 5.5 as the LLM baseline
  export TYPESAFE_API_KEY=...  ANTHROPIC_API_KEY=...
  python run.py

  # Jev through Cloudflare Workers AI instead
  export CLOUDFLARE_ACCOUNT_ID=...  CLOUDFLARE_API_TOKEN=...
  python run.py --jev-backend cloudflare

  # A GPT model as the baseline (set its price in jevbench/config.py)
  export OPENAI_API_KEY=...
  python run.py --llm-backend openai --llm-model gpt-...

  # Everything on Cloudflare (free daily allocation): Jev + Llama 3.3 70B as the LLM baseline
  export CLOUDFLARE_ACCOUNT_ID=...  CLOUDFLARE_API_TOKEN=...
  python run.py --jev-backend cloudflare --llm-backend cloudflare

  # Free LLM on Groq (no card): export GROQ_API_KEY=...
  python run.py --jev-backend cloudflare --llm-backend groq

  # Test the code with no keys and no network (fake models, numbers are meaningless)
  python run.py --dry-run

  # Only rebuild the report from saved results (no API calls)
  python run.py --report-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jevbench import report as R
from jevbench.config import TARGET_ACCURACY
from jevbench.jev import make_jev
from jevbench.llm import make_llm
from jevbench.questions import QUESTIONS_VERSION
from jevbench.runner import run_system

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev-backend", default="typesafe", choices=["typesafe", "cloudflare"])
    ap.add_argument("--jev-model", default=None, help="default jev-1.13.0 (TypeSafe) / typesafe/jev (Cloudflare)")
    ap.add_argument("--llm-backend", default="anthropic", choices=["anthropic", "openai", "cloudflare", "groq"])
    ap.add_argument("--llm-model", default=None, help="default claude-sonnet-5-5")
    ap.add_argument("--dry-run", action="store_true", help="fake models, offline; for testing the code only")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = HERE / ("results_dryrun" if args.dry_run else "results")
    out.mkdir(exist_ok=True)
    tickets = pd.read_csv(HERE / "data" / "tickets.csv")
    tricky = pd.read_csv(HERE / "data" / "tricky.csv")

    if not args.report_only:
        jev = make_jev("fake" if args.dry_run else args.jev_backend, args.jev_model)
        llm = make_llm("fake" if args.dry_run else args.llm_backend, args.llm_model)
        try:
            for name, df in (("tickets", tickets), ("tricky", tricky)):
                run_system(jev, df, out / f"jev_{name}.jsonl", verbose=not args.quiet)
                run_system(llm, df, out / f"llm_{name}.jsonl", verbose=not args.quiet)
        finally:
            jev.close()
            llm.close()

    build_report(out, tickets, tricky, dry_run=args.dry_run)


def _model(df: pd.DataFrame) -> str:
    m = df["model"].dropna()
    return str(m.iloc[0]) if len(m) else "no successful calls"


def build_report(out: Path, tickets: pd.DataFrame, tricky: pd.DataFrame, dry_run: bool = False) -> dict:
    jev_recs = R.load_results(out / "jev_tickets.jsonl")
    llm_recs = R.load_results(out / "llm_tickets.jsonl")
    jev_df, llm_df = R.to_frame(jev_recs, tickets), R.to_frame(llm_recs, tickets)

    # Results table ------------------------------------------------------------
    table = R.results_table([R.summarize(jev_df, "jev"), R.summarize(llm_df, "llm")])
    table.to_csv(out / "results_table.csv")
    pretty = R.format_table(table)
    langs = R.per_language({"Jev": jev_df, "LLM": llm_df})

    # Confidence threshold -------------------------------------------------------
    jev_name = f"Jev ({_model(jev_df)})"
    llm_name = f"LLM self-reported ({_model(llm_df)})"
    curves = {jev_name: R.threshold_curve(jev_df), llm_name: R.threshold_curve(llm_df)}
    points = {k: R.operating_point(c) for k, c in curves.items()}
    pd.concat({k: c for k, c in curves.items()}, names=["system"]).to_csv(out / "threshold_curve.csv")
    R.plot_thresholds(curves, points, out / "confidence_threshold.png",
                      title_note="DRY RUN, FAKE DATA" if dry_run else "")

    # Tricky cases ----------------------------------------------------------------
    jev_t_recs = R.load_results(out / "jev_tricky.jsonl")
    jev_t = R.to_frame(jev_t_recs, tricky)
    llm_t = R.to_frame(R.load_results(out / "llm_tricky.jsonl"), tricky)
    tt = R.tricky_table(jev_t, jev_t_recs, llm_t)
    tt.to_csv(out / "tricky_cases.csv", index=False)

    # Errors / disagreements for error analysis ---------------------------------
    errs = jev_df[~jev_df["queue_ok"]][["id", "text", "queue", "pred_queue", "confidence"]]
    llm_bad = llm_df[~llm_df["valid"]][["id", "status", "raw", "error"]]

    # Markdown report ---------------------------------------------------------------
    lines = []
    if dry_run:
        lines += ["> **DRY RUN — fake models. These numbers are meaningless; they only show the pipeline works.**", ""]
    lines += [f"# Results (questions {QUESTIONS_VERSION})", "",
              f"{len(tickets)} labeled tickets, {len(tricky)} tricky inputs. One call per ticket, sequential, "
              "latency = wall-clock time of the API call from the machine that ran `run.py`.", "",
              "## Jev vs LLM", "", R.to_markdown(pretty, index=True), "",
              "Invalid = API error, unparseable output, or a label outside the allowed set. "
              "'Recovered JSON' = the LLM wrapped its JSON in fences or prose but it could still be extracted "
              "(counted as valid).", "",
              "### Queue accuracy by language", "", R.to_markdown(langs.map(lambda v: f'{v:.0%}' if v <= 1 else int(v)), index=True), "",
              "## Confidence threshold", "", "![confidence threshold](confidence_threshold.png)", ""]
    for name, p in points.items():
        if p:
            n = len(tickets)
            lines.append(f"- **{name}**: at threshold **{p['threshold']:.2f}**, {p['auto_n']}/{n} tickets "
                         f"(**{p['coverage']:.0%}**) are auto-decided with {p['accuracy']:.1%} accuracy "
                         f"(95% Wilson lower bound {p['acc_wilson_low']:.1%}); "
                         f"**{n - p['auto_n']} ({1 - p['coverage']:.0%}) go to a human**.")
        else:
            lines.append(f"- **{name}**: no threshold reaches {TARGET_ACCURACY:.0%} accuracy.")
    lines += ["", "## Tricky inputs", "", R.to_markdown(tt), "",
              "## Jev routing errors on the labeled set", "", R.to_markdown(errs) if len(errs) else "None.", "",
              "## LLM invalid outputs", "", R.to_markdown(llm_bad) if len(llm_bad) else "None.", ""]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + R.to_markdown(pretty, index=True))
    for name, p in points.items():
        print(name, "->", p)
    print(f"\nWrote {out}/results.md, results_table.csv, confidence_threshold.png, tricky_cases.csv")
    return {"table": pretty, "points": points, "tricky": tt, "langs": langs}


if __name__ == "__main__":
    main()
