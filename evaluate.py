"""
SupportDesk evaluation entry point.

Scores retrieval source-coverage @ top-k and response behavior / citations
against dataset/eval_questions.jsonl (held under src/dataset/).

Usage (from repo root, with Qdrant already ingested):

    uv run python evaluate.py
    uv run python evaluate.py --eval-file src/dataset/eval_questions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Support both `from config` and `from src.*` import styles used in this repo.
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    db_api_key,
    db_url,
    logger,
)
from llm import LLMService  # noqa: E402
from query import extract_evidence  # noqa: E402
from settings import initialize_settings  # noqa: E402
from vector_store import initialize_qdrant_index, retrieve_nodes  # noqa: E402


DEFAULT_EVAL_FILE = SRC / "dataset" / "eval_questions.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "results"
RESULTS_JSONL = "eval_results.jsonl"
RESULTS_SUMMARY = "evaluation_summary.md"

BEHAVIOR_TO_STATUS = {
    "answer": "answered",
    "clarify": "needs_clarification",
    "abstain": "insufficient_evidence",
}

STATUS_ALIASES = {
    "answered": "answered",
    "answer": "answered",
    "needs_clarification": "needs_clarification",
    "need_clarification": "needs_clarification",
    "clarify": "needs_clarification",
    "insufficient_evidence": "insufficient_evidence",
    "abstain": "insufficient_evidence",
}

EVIDENCE_ID_RE = re.compile(r"\bE\d+\b", re.IGNORECASE)


def normalize_filename(name: str | None) -> str:
    if not name:
        return ""
    return Path(str(name).strip().replace("\\", "/")).name.lower()


def normalize_status(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return STATUS_ALIASES.get(key, key)


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
            cases.append(row)

    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def build_llm() -> LLMService:
    if LLM_PROVIDER == "ollama":
        model_options = {"base_url": OLLAMA_HOST, "temperature": 0}
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Set OPENAI_API_KEY in .env")
        model_options = {"api_key": OPENAI_API_KEY}
    else:
        raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")

    return LLMService(
        provider=LLM_PROVIDER,
        model_name=LLM_MODEL,
        model_options=model_options,
    )


def score_retrieval(
    expected_sources: list[str],
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[bool | None, list[str]]:
    """File-level all-required-source coverage at top-k.

    Returns (pass, retrieved_filenames). pass is None when no sources expected.
    """
    retrieved_files = sorted(
        {
            normalize_filename(chunk.get("filename"))
            for chunk in retrieved_chunks
            if normalize_filename(chunk.get("filename"))
        }
    )

    expected = [normalize_filename(name) for name in expected_sources if name]
    if not expected:
        return None, retrieved_files

    retrieval_pass = all(name in retrieved_files for name in expected)
    return retrieval_pass, retrieved_files


def extract_cited_ids(decision: Any) -> list[str]:
    raw_parts: list[str] = []
    if decision is None:
        return []

    evidence_id = getattr(decision, "evidence_id", None)
    message = getattr(decision, "message", None)
    if evidence_id:
        raw_parts.append(str(evidence_id))
    if message:
        raw_parts.append(str(message))

    found = EVIDENCE_ID_RE.findall(" ".join(raw_parts))
    # Preserve order, unique, canonical uppercase (E1, E2, ...)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in found:
        key = item.upper()
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def score_citations(
    decision: Any,
    retrieved_chunks: list[dict[str, Any]],
    predicted_status: str | None,
) -> tuple[bool, list[dict[str, Any]]]:
    valid_ids = {
        str(chunk.get("evidence_id", "")).upper(): chunk
        for chunk in retrieved_chunks
        if chunk.get("evidence_id")
    }
    cited_ids = extract_cited_ids(decision)

    mapping: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for citation_id in cited_ids:
        chunk = valid_ids.get(citation_id)
        if chunk is None:
            unresolved.append(citation_id)
            mapping.append(
                {
                    "citation_id": citation_id,
                    "filename": None,
                    "source_id": None,
                    "exact_passage": None,
                    "resolved": False,
                }
            )
            continue

        mapping.append(
            {
                "citation_id": citation_id,
                "filename": chunk.get("filename"),
                "source_id": chunk.get("chunk_id"),
                "exact_passage": chunk.get("exact_text"),
                "resolved": True,
            }
        )

    # Abstentions may omit citations; answers should cite real evidence.
    if predicted_status == "insufficient_evidence":
        citation_validity_pass = len(unresolved) == 0
    elif predicted_status == "answered":
        citation_validity_pass = bool(cited_ids) and len(unresolved) == 0
    else:
        # clarification: citations optional, but must not invent IDs
        citation_validity_pass = len(unresolved) == 0

    return citation_validity_pass, mapping


async def evaluate_one(
    *,
    case: dict[str, Any],
    index,
    llm: LLMService,
    run_id: str,
    top_k: int,
) -> dict[str, Any]:
    question_id = case.get("id")
    question = case.get("question", "")
    expected_behavior = case.get("expected_behavior")
    expected_sources = case.get("expected_sources") or []
    expected_status = normalize_status(BEHAVIOR_TO_STATUS.get(expected_behavior, expected_behavior))

    failure_stages: list[str] = []
    predicted_status: str | None = None
    generated_answer: str | None = None
    decision = None
    citation_mapping: list[dict[str, Any]] = []
    citation_validity_pass = False
    behavior_pass = False
    retrieval_pass: bool | None = None
    retrieved_chunks: list[dict[str, Any]] = []
    retrieved_files: list[str] = []

    total_start = time.perf_counter()

    try:
        nodes = await asyncio.to_thread(retrieve_nodes, index, question, top_k)
        evidence = extract_evidence(nodes)
        retrieved_chunks = [
            {
                "rank": idx,
                "filename": item.get("filename"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "text": item.get("exact_text"),
                "evidence_id": item.get("evidence_id"),
            }
            for idx, item in enumerate(evidence, start=1)
        ]

        retrieval_pass, retrieved_files = score_retrieval(expected_sources, evidence)
    except Exception as exc:
        failure_stages.append("retrieval")
        logger.exception("Retrieval failed for %s", question_id)
        elapsed_ms = int((time.perf_counter() - total_start) * 1000)
        return {
            "run_id": run_id,
            "question_id": question_id,
            "question": question,
            "expected_sources": expected_sources,
            "expected_status": expected_status,
            "retrieved_chunks": [],
            "retrieved_files": [],
            "retrieval_pass": None if not expected_sources else False,
            "generated_answer": None,
            "predicted_status": None,
            "citation_mapping": [],
            "correctness_pass": None,
            "grounding_pass": None,
            "citation_validity_pass": False,
            "citation_support_pass": None,
            "behavior_pass": False,
            "elapsed_ms": elapsed_ms,
            "failure_stages": failure_stages,
            "reviewer_notes": f"retrieval error: {exc}",
        }

    try:
        decision = await llm.generate_decision(
            query=question,
            evidence=evidence,
            on_token=None,
        )
        predicted_status = normalize_status(getattr(decision, "status", None))
        generated_answer = getattr(decision, "message", None)
        behavior_pass = predicted_status == expected_status
        citation_validity_pass, citation_mapping = score_citations(
            decision,
            evidence,
            predicted_status,
        )
    except Exception as exc:
        failure_stages.append("generation")
        logger.exception("Generation failed for %s", question_id)
        generated_answer = None
        predicted_status = None
        behavior_pass = False
        citation_validity_pass = False
        citation_mapping = []
        reviewer_error = f"generation error: {exc}"
    else:
        reviewer_error = ""

    elapsed_ms = int((time.perf_counter() - total_start) * 1000)

    return {
        "run_id": run_id,
        "question_id": question_id,
        "question": question,
        "expected_sources": expected_sources,
        "expected_status": expected_status,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_files": retrieved_files,
        "retrieval_pass": retrieval_pass,
        "generated_answer": generated_answer,
        "predicted_status": predicted_status,
        "citation_mapping": citation_mapping,
        "correctness_pass": None,  # manual review
        "grounding_pass": None,  # manual review
        "citation_validity_pass": citation_validity_pass,
        "citation_support_pass": None,  # manual review
        "behavior_pass": behavior_pass,
        "elapsed_ms": elapsed_ms,
        "failure_stages": failure_stages,
        "reviewer_notes": reviewer_error,
        "reference_answer": case.get("expected_answer"),
    }


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(results: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    labeled = [row for row in results if row.get("expected_sources")]
    retrieval_scored = [row for row in labeled if row.get("retrieval_pass") is not None]
    retrieval_passes = sum(1 for row in retrieval_scored if row.get("retrieval_pass"))
    behavior_passes = sum(1 for row in results if row.get("behavior_pass"))
    citation_passes = sum(1 for row in results if row.get("citation_validity_pass"))

    latencies = [int(row["elapsed_ms"]) for row in results if row.get("elapsed_ms") is not None]

    special_ids = {"T13", "T14", "T15"}
    special = {
        row["question_id"]: {
            "expected_status": row.get("expected_status"),
            "predicted_status": row.get("predicted_status"),
            "behavior_pass": row.get("behavior_pass"),
        }
        for row in results
        if row.get("question_id") in special_ids
    }

    failures = [
        {
            "question_id": row.get("question_id"),
            "retrieval_pass": row.get("retrieval_pass"),
            "behavior_pass": row.get("behavior_pass"),
            "citation_validity_pass": row.get("citation_validity_pass"),
            "failure_stages": row.get("failure_stages"),
            "predicted_status": row.get("predicted_status"),
            "expected_status": row.get("expected_status"),
            "retrieved_files": row.get("retrieved_files"),
            "expected_sources": row.get("expected_sources"),
        }
        for row in results
        if (row.get("retrieval_pass") is False)
        or (not row.get("behavior_pass"))
        or (not row.get("citation_validity_pass"))
        or row.get("failure_stages")
    ]

    return {
        **meta,
        "total_cases": len(results),
        "source_labeled_cases": len(retrieval_scored),
        "all_required_source_coverage_at_3": {
            "pass_count": retrieval_passes,
            "total": len(retrieval_scored),
            "ratio": (retrieval_passes / len(retrieval_scored)) if retrieval_scored else None,
            "target": "at least 13/14",
        },
        "behavior_accuracy": {
            "pass_count": behavior_passes,
            "total": len(results),
            "ratio": (behavior_passes / len(results)) if results else None,
        },
        "citation_validity": {
            "pass_count": citation_passes,
            "total": len(results),
            "ratio": (citation_passes / len(results)) if results else None,
        },
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "mean": statistics.mean(latencies) if latencies else None,
        },
        "special_cases": special,
        "failures": failures,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_summary_md(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cov = summary["all_required_source_coverage_at_3"]
    beh = summary["behavior_accuracy"]
    cit = summary["citation_validity"]
    lat = summary["latency_ms"]

    lines = [
        "# Evaluation summary",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Timestamp (UTC): `{summary.get('timestamp_utc')}`",
        f"- Embedding model: `{summary.get('embedding_model')}`",
        f"- LLM provider/model: `{summary.get('llm_provider')}` / `{summary.get('llm_model')}`",
        f"- Top-k: `{summary.get('top_k')}`",
        f"- Chunk size / overlap: `{summary.get('chunk_size')}` / `{summary.get('chunk_overlap')}`",
        f"- Eval file: `{summary.get('eval_file')}`",
        "",
        "## Scores",
        "",
        "| Measure | Target | Actual |",
        "| --- | --- | --- |",
        (
            f"| All-required-source coverage at 3 | At least 13/14 | "
            f"{cov['pass_count']}/{cov['total']} "
            f"({(cov['ratio'] or 0):.2%}) |"
        ),
        (
            f"| Behavior accuracy | All 16 reviewed | "
            f"{beh['pass_count']}/{beh['total']} "
            f"({(beh['ratio'] or 0):.2%}) |"
        ),
        (
            f"| Citation validity | No invented/unresolved citations | "
            f"{cit['pass_count']}/{cit['total']} "
            f"({(cit['ratio'] or 0):.2%}) |"
        ),
        (
            f"| Latency p50 / p95 (ms) | n/a | "
            f"{lat['p50']:.0f} / {lat['p95']:.0f} |"
            if lat["p50"] is not None and lat["p95"] is not None
            else "| Latency p50 / p95 (ms) | n/a | n/a |"
        ),
        "",
        "## Special cases (T13 / T14 / T15)",
        "",
    ]

    for question_id, payload in summary.get("special_cases", {}).items():
        lines.append(
            f"- `{question_id}`: expected `{payload.get('expected_status')}`, "
            f"predicted `{payload.get('predicted_status')}`, "
            f"pass={payload.get('behavior_pass')}"
        )

    lines.extend(["", "## Per-question results", ""])
    lines.append(
        "| id | retrieval | behavior | citations | expected | predicted | ms |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in results:
        retrieval = row.get("retrieval_pass")
        retrieval_cell = "n/a" if retrieval is None else ("pass" if retrieval else "fail")
        lines.append(
            "| {id} | {retrieval} | {behavior} | {citations} | {expected} | {predicted} | {ms} |".format(
                id=row.get("question_id"),
                retrieval=retrieval_cell,
                behavior="pass" if row.get("behavior_pass") else "fail",
                citations="pass" if row.get("citation_validity_pass") else "fail",
                expected=row.get("expected_status"),
                predicted=row.get("predicted_status"),
                ms=row.get("elapsed_ms"),
            )
        )

    lines.extend(["", "## Failure analysis", ""])
    failures = summary.get("failures") or []
    if not failures:
        lines.append("No automated failures detected.")
    else:
        for item in failures:
            lines.append(
                (
                    f"- `{item.get('question_id')}`: retrieval={item.get('retrieval_pass')}, "
                    f"behavior={item.get('behavior_pass')}, "
                    f"citations={item.get('citation_validity_pass')}, "
                    f"expected_sources={item.get('expected_sources')}, "
                    f"retrieved_files={item.get('retrieved_files')}, "
                    f"stages={item.get('failure_stages')}"
                )
            )

    lines.extend(
        [
            "",
            "## Manual review still required",
            "",
            "Fields left null for human review: `correctness_pass`, "
            "`grounding_pass`, `citation_support_pass`.",
            "Check that answers preserve policy conditions/exceptions and that "
            "cited passages actually support each claim.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_table(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    for row in results:
        retrieval = row.get("retrieval_pass")
        retrieval_cell = "n/a" if retrieval is None else ("Y" if retrieval else "N")
        print(
            f"{row.get('question_id')}: "
            f"retrieval={retrieval_cell} "
            f"behavior={'Y' if row.get('behavior_pass') else 'N'} "
            f"citations={'Y' if row.get('citation_validity_pass') else 'N'} "
            f"pred={row.get('predicted_status')} "
            f"exp={row.get('expected_status')} "
            f"ms={row.get('elapsed_ms')}"
        )

    cov = summary["all_required_source_coverage_at_3"]
    beh = summary["behavior_accuracy"]
    cit = summary["citation_validity"]
    lat = summary["latency_ms"]
    print("-" * 72)
    print(
        f"Source coverage @3: {cov['pass_count']}/{cov['total']} "
        f"({(cov['ratio'] or 0):.2%})"
    )
    print(
        f"Behavior accuracy:  {beh['pass_count']}/{beh['total']} "
        f"({(beh['ratio'] or 0):.2%})"
    )
    print(
        f"Citation validity:  {cit['pass_count']}/{cit['total']} "
        f"({(cit['ratio'] or 0):.2%})"
    )
    if lat["p50"] is not None:
        print(f"Latency p50/p95:    {lat['p50']:.0f} / {lat['p95']:.0f} ms")
    print("=" * 72)


async def run_evaluation(eval_file: Path, results_dir: Path, top_k: int) -> dict[str, Any]:
    if not db_url:
        raise RuntimeError("Set DB_URL in .env before running evaluation")

    cases = load_eval_cases(eval_file)
    run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    initialize_settings()
    resources = initialize_qdrant_index(
        url=db_url,
        api_key=db_api_key,
        collection=COLLECTION_NAME,
        timeout=30,
        prefer_grpc=False,
    )
    llm = build_llm()

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            question_id = case.get("id")
            print(f"\n>>> Evaluating {question_id} ...")
            row = await evaluate_one(
                case=case,
                index=resources["index"],
                llm=llm,
                run_id=run_id,
                top_k=top_k,
            )
            results.append(row)
    finally:
        resources["client"].close()

    from config import CHUNK_OVERLAP, CHUNK_SIZE

    meta = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "top_k": top_k,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "eval_file": str(eval_file),
        "collection": COLLECTION_NAME,
    }
    summary = summarize(results, meta)

    jsonl_path = results_dir / RESULTS_JSONL
    summary_path = results_dir / RESULTS_SUMMARY
    write_jsonl(jsonl_path, results)
    write_summary_md(summary_path, summary, results)

    # Machine-readable summary alongside the markdown report.
    summary_json_path = results_dir / "evaluation_summary.json"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print_console_table(results, summary)
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_json_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SupportDesk RAG quality")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Path to eval questions JSONL",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for eval outputs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Retrieval top-k (default from config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(
            run_evaluation(
                eval_file=args.eval_file.resolve(),
                results_dir=args.results_dir.resolve(),
                top_k=args.top_k,
            )
        )
    except Exception as exc:
        logger.exception("Evaluation failed")
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
