"""Build a retrospective experiment report from existing logs/artifacts only.

Run from any directory with the ordinary Python interpreter. No GPU operations,
training, remote writes, or modifications to experiment results are performed.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "docs/conversation-experiments-01a10259.md"
MARKER = "<!-- GENERATED APPENDICES -->"
CID = "01a10259-7a3b-7e00-b3cb-d7f305496a7f"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, label=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return f"[{label or p.name}](<{p}>)"


def fence(obj):
    return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```\n"


def cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    ) + "\n\n"


def compact(value):
    """Retain scalar/nested metadata; identify large trace arrays explicitly."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= 16 and all(not isinstance(v, (dict, list)) for v in value):
            return value
        return {"array_length": len(value), "full_values": "See original linked artifact"}
    return value


def family(name):
    rules = [
        ("latent_body_progress", "E32"), ("latent_context_flow", "E32"),
        ("latent_allowance", "E31"), ("latent_objective", "E28"),
        ("latent_three_term", "E28"), ("latent_count_toys", "E26"),
        ("latent_count_ablation", "E27"), ("latent_count_candidate", "E27"),
        ("latent_count_arithmetic", "E27"), ("latent_count_humaneval", "E27"),
        ("latent_count_singleton", "E27"), ("latent_pca_flow", "E25"),
        ("latent_flow", "E25"), ("latent_survivors", "E24"),
        ("latent_spectral_survivors", "E24"), ("latent_residual", "E24"),
        ("latent_full", "E23"), ("latent_consistency", "E22"),
        ("latent_attention", "E21"), ("latent_information_depth", "E20"),
        ("context_reconstruction", "E19"), ("latent_depth", "E18"),
        ("latent_conditioning", "E17"), ("latent_support_count", "E16"),
        ("latent_information", "E15"), ("latent_finite", "E14"),
        ("latent_gradient", "E13"), ("gradient_", "E13"),
        ("posterior_and", "E12"), ("latent_posterior", "E12"),
        ("late_count", "E12"), ("latent_value", "E11"),
        ("reveal_value", "E11"), ("body_gate", "E10"),
        ("content_", "E10"), ("eb_singleton", "E10"),
        ("latent_risk", "E09"), ("learned_risk", "E09"),
        ("latent_support_factorial", "E08"), ("latent_factorial", "E07"),
        ("latent_count_verified", "E07"), ("latent_arithmetic_count", "E07"),
        ("latent_stability", "E06"), ("latent_current", "E06"),
        ("latent_humaneval20_g256", "E05"), ("latent_humaneval", "E04–E06"),
        ("latent_arithmetic", "E04/E06/E11/E12"),
        ("infogain_lab", "E02/E03/E13"), ("ugc_", "E26"),
        ("latent_only", "E24/E25"), ("scheduler_scoreboard", "Cross-study index"),
    ]
    for prefix, tag in rules:
        if name.startswith(prefix):
            return tag
    return "Other preserved diagnostic"


def flatten_output(output):
    if isinstance(output, list):
        return "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in output)
    return str(output)


def load_conversation():
    sources, messages, calls, outputs = [], [], [], {}
    paths = sorted(Path("/root/.codex/sessions").rglob(f"*{CID}*.jsonl"))
    assert len(paths) == 2, f"Expected two conversation streams, found {len(paths)}"
    for n, path in enumerate(paths, 1):
        lines = path.read_text().splitlines()
        sources.append({"label": f"S{n}", "path": str(path), "sha256": digest(path), "records": len(lines)})
        for line, text in enumerate(lines, 1):
            row = json.loads(text)
            q = row.get("payload", {})
            ref = f"S{n}:L{line}"
            stamp = row.get("timestamp")
            if row["type"] != "response_item":
                continue
            if q.get("type") == "message":
                body = "\n".join(c.get("text", "") for c in q.get("content", []))
                if q.get("role") in ("user", "assistant") and q.get("channel") != "analysis" and not body.startswith(
                    ("<environment_context>", "<external_codex_apps_open_page>", "<codex_internal_context")
                ):
                    messages.append({"ref": ref, "time": stamp, "role": q.get("role"), "text": body})
            elif q.get("type") in ("function_call", "custom_tool_call"):
                calls.append({"ref": ref, "time": stamp, "name": q.get("name"), "id": q.get("call_id"), "arguments": q.get("arguments", q.get("input", ""))})
            elif q.get("type") in ("function_call_output", "custom_tool_call_output"):
                outputs[q.get("call_id")] = {"ref": ref, "output": q.get("output")}
    return sources, messages, calls, outputs


@lru_cache(maxsize=None)
def root_metadata(root):
    parent = ROOT / "outputs" / root
    result = {}
    for name in ("manifest.json", "results.json"):
        path = parent / name
        if path.exists():
            data = read_json(path)
            if isinstance(data, dict):
                result[name] = {k: compact(v) for k, v in data.items() if k not in ("samples", "summaries", "scheduler_metrics")}
    return result


def status(path, method):
    root = path.parts[1]
    if root == "latent_context_flow_arithmetic" and "flow" in method:
        return "EXCLUDED: initial hook omitted residual derivative; corrected rerun exists"
    if root == "latent_body_progress_humaneval20" and "flow" in method:
        return "EXCLUDED: initial hook omitted residual derivative; corrected rerun exists"
    if root == "latent_count_arithmetic":
        return "Scored arm in interrupted first suite; diagnostic adapter failure followed"
    if root == "latent_full_humaneval164":
        return "Completed frozen full-164 evaluation"
    if "humaneval" in root or "verified20" in root or root in ("latent_count_singleton20", "latent_pca_flow_humaneval20_lean"):
        return "Completed standalone generation arm"
    if root.endswith(("80", "100", "160", "60")) or root in ("latent_factorial", "posterior_and_count160"):
        return "Completed packed exploratory arm; use same-batch controls"
    return "Completed scored exploratory arm / repeated control"


def build_catalog(calls, outputs):
    catalog = []
    call_text = [(c, str(c["arguments"]) + "\n" + flatten_output(outputs.get(c["id"], {}).get("output", ""))) for c in calls]
    root_references = {}
    files = sorted((ROOT / "outputs").rglob("*.json"))
    for path in files:
        rel = path.relative_to(ROOT)
        if rel.parts[1].startswith("llada_") or rel.parts[1] == "schedulerlens":
            continue
        data = read_json(path)
        if not (isinstance(data, dict) and isinstance(data.get("samples"), list) and ("metrics" in data or "summary" in data)):
            continue
        metrics = data.get("metrics", data.get("summary", {}))
        method = metrics.get("variant", metrics.get("method", metrics.get("name", metrics.get("cell", data.get("run", path.parent.name if path.name == "results.json" else path.stem)))))
        config = {k: compact(v) for k, v in data.items() if k not in ("metrics", "summary", "samples", "scheduler_metrics", "batch_metrics", "selected_counts")}
        sched = data.get("scheduler_metrics", {})
        n = metrics.get("samples", len(data["samples"]))
        seed = metrics.get("seed", data.get("seed", (data.get("config") or {}).get("seed")))
        if seed is None:
            m = re.search(r"/seed(\d+)/", str(rel))
            if m:
                seed = int(m.group(1))
        if seed is None and "arithmetic" in rel.parts[1]:
            m = re.search(r"seed(\d+)$", rel.parts[1])
            if m:
                seed = int(m.group(1))
        fwd = metrics.get("forward_evaluations_per_sample", metrics.get("forward_evaluations_per_canvas"))
        bwd = metrics.get("backward_evaluations_per_sample", metrics.get("backward_evaluations_per_canvas"))
        pbwd = metrics.get("partial_backward_evaluations_per_sample", metrics.get("partial_backward_evaluations_per_canvas"))
        work_source = "recorded normalized metric"
        if fwd is None and n:
            seq = metrics.get("sequence_evaluations", sched.get("sequence_evaluations"))
            if seq is not None:
                fwd = seq / n
                work_source = "derived: recorded sequence_evaluations / samples"
        if bwd is None and n:
            seq = metrics.get("backward_sequence_evaluations", sched.get("backward_sequence_evaluations"))
            if seq is not None:
                bwd = seq / n
        if pbwd is None and n:
            seq = sched.get("partial_backward_sequence_evaluations")
            if seq is not None:
                pbwd = seq / n
        root = rel.parts[1]
        metadata = root_metadata(root)
        # Packed arms sometimes store only outcome/count summaries. The group's
        # average forward work is valid only when uniform normal work is recorded.
        if fwd is None and "results.json" in metadata:
            aggregate = read_json(ROOT / "outputs" / root / "results.json")
            work = aggregate.get("scheduler_metrics", {})
            total_n = aggregate.get("total_batch_size")
            if total_n and work.get("sequence_evaluations") is not None and not work.get("backward_sequence_evaluations", 0):
                steps = aggregate.get("steps")
                if work["sequence_evaluations"] == total_n * steps:
                    fwd = float(steps)
                    bwd = 0.0
                    work_source = "derived: uniform normal forwards from packed aggregate"
        if root not in root_references:
            root_references[root] = [dict(ref=c["ref"], time=c["time"]) for c, text in call_text if root in text]
        references = root_references[root]
        samples = data["samples"]
        task_outcomes = [{k: s[k] for k in ("task_id", "id", "passed", "result", "unfinished_masks", "slot_correct", "exact_slot_correct", "correct_slots", "correct_worksheets", "all_correct") if k in s} for s in samples]
        if samples and "task_id" in samples[0] and "passed" in samples[0]:
            observed = sum(bool(s.get("passed")) for s in samples)
            assert metrics.get("passed", observed) == observed, f"Outcome mismatch: {rel}"
        catalog.append({
            "arm_id": f"A{len(catalog)+1:04d}", "family": family(root), "study": root,
            "method": method, "seed": seed, "samples": n, "status": status(rel, method),
            "artifact": str(rel), "artifact_sha256": digest(path),
            "configuration_and_provenance": config, "study_metadata": metadata,
            "metrics": metrics, "scheduler_scalar_metadata": {k: compact(v) for k, v in sched.items() if not isinstance(v, list)},
            "trace_arrays": {k: len(v) for k, v in sched.items() if isinstance(v, list)},
            "forward_evaluations_per_sample": fwd, "backward_evaluations_per_sample": bwd,
            "partial_backward_evaluations_per_sample": pbwd, "work_metric_source": work_source,
            "related_conversation_records": references, "task_outcomes": task_outcomes,
        })
    return catalog


def parse_commands(calls, outputs):
    commands = []
    for c in calls:
        arguments = str(c["arguments"])
        for quoted in re.findall(r"\bcmd\s*:\s*(\"(?:\\.|[^\"\\])*\")", arguments):
            try:
                command = json.loads(quoted)
            except json.JSONDecodeError:
                continue
            if not any(t in command for t in ("python", "pytest", "unittest", "latent", "infogain", "spectral", "rocm", "evaluator")):
                continue
            category = "inspection / result analysis"
            if "unittest" in command or "pytest" in command:
                category = "implementation validation"
            elif "python - <<" in command or "python3 - <<" in command or "bin/python - <<" in command:
                category = "inline computation / diagnostic / inspection"
            elif any(t in command for t in ("_bench", "infogain_collect", "train_latent", "latent_stability_sweep", "latent_gradient_sweep")):
                category = "experiment launch / collection / training"
            elif "audit" in command or "_report" in command or "budget_replay" in command:
                category = "audit / reporting / retrospective evaluation"
            out = outputs.get(c["id"], {})
            commands.append({"command_id": f"C{len(commands)+1:04d}", "ref": c["ref"], "time": c["time"], "category": category, "command": command, "output_ref": out.get("ref")})
    return commands


def scan_auxiliary(roots):
    items = []
    for root in sorted(roots):
        for path in sorted((ROOT / "outputs" / root).rglob("*")):
            if not path.is_file() or path.suffix not in (".json", ".csv", ".md", ".py", ".log", ".jsonl"):
                continue
            items.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path), "family": family(root)})
    return items


def appendices(sources, messages, catalog, commands, artifacts):
    out = ["\n## Appendix A. Coverage, evidence, and inventory conventions\n\n"]
    out.append(f"The inventory contains **{len(catalog)} scored arm artifacts across {len(set(a['study'] for a in catalog))} study directories**, {len(commands)} recovered relevant shell commands, and {len(messages)} public conversation messages. These include repeats and excluded diagnostic records. Raw traces/completions remain in their linked artifacts. No samples are counted as new experiments simply because they appear in a checkpoint and a final result.\n\n")
    out.append(table(["Stream", "Records", "Source", "SHA-256"], [(s["label"], s["records"], link(s["path"]), s["sha256"]) for s in sources]))
    out.append("Supplementary machine-readable files: " + ", ".join(link(HERE / name) for name in ("experiment_catalog.json", "experiment_arms.csv", "sample_outcomes.csv", "commands.json", "artifact_manifest.json", "coverage.json")) + ".\n\n")
    out.append("Table forward/backward values are sequence evaluations **per sample**, not model-call counts. A dash means the value was not recoverable as a normalized metric. Generation seconds/memory are preserved in exact per-arm metrics; packed shared times are not misreported as individual arm timing. Recorded steps may be a cap; `steps_taken` in details gives actual work where available. Large per-step trace arrays are linked and identified by length instead of printed in full. Historical eligibility flags are retained as evidence but should be read against the final constraint explained in Section 3.\n\n")
    out.append("## Appendix B. Every surviving scored arm, grouped by study\n\n")
    studies = defaultdict(list)
    for a in catalog:
        studies[a["study"]].append(a)
    for root, arms in sorted(studies.items(), key=lambda item: (min((r["time"] for a in item[1] for r in a["related_conversation_records"]), default="9999"), item[0])):
        out.append(f"### {root}\n\nResearch family: **{family(root)}**. {len(arms)} saved scored arms.\n\n")
        refs = sorted({(r["time"], r["ref"]) for a in arms for r in a["related_conversation_records"]})
        if refs:
            out.append(f"First recovered related record: `{refs[0][1]}` at {refs[0][0]}. This can be a launch, a source edit, or a later inspection; it is not an inferred completion timestamp.\n\n")
        out.append(table(["Arm", "Method", "Seed", "N", "Result", "Worksheet", "Fwd", "Full bwd", "Partial bwd", "Masks", "Status"], [
            (f"[{a['arm_id']}](#arm-{a['arm_id'].lower()})", a["method"], a["seed"], a["samples"],
             f"{a['metrics']['passed']}/{a['samples']}" if "passed" in a["metrics"] else f"{100*a['metrics'].get('slot_accuracy', 0):.4f}% slots",
             f"{100*a['metrics']['worksheet_accuracy']:.4f}%" if "worksheet_accuracy" in a["metrics"] else None,
             a["forward_evaluations_per_sample"], a["backward_evaluations_per_sample"], a["partial_backward_evaluations_per_sample"],
             a["metrics"].get("unfinished_masks"), "EXCLUDED" if a["status"].startswith("EXCLUDED") else "Scored") for a in arms]))
        metadata = root_metadata(root)
        if metadata:
            out.append("<details>\n<summary>Shared study configuration, source hashes, and protocol</summary>\n\n" + fence(metadata) + "\n</details>\n\n")
        for a in arms:
            out.append(f"<a id=\"arm-{a['arm_id'].lower()}\"></a>\n\n<details>\n<summary>{a['arm_id']} — {html.escape(a['method'])}, seed {a['seed']} — exact settings and results</summary>\n\n")
            out.append(f"**Status:** {a['status']}. **Source:** {link(a['artifact'], a['artifact'])}.\n\n")
            out.append(f"**Artifact SHA-256:** `{a['artifact_sha256']}`. **Compute normalization:** {a['work_metric_source']}.\n\n")
            exact = {k: a[k] for k in ("configuration_and_provenance", "metrics", "scheduler_scalar_metadata", "trace_arrays")}
            out.append(fence(exact))
            if a["task_outcomes"] and "task_id" in a["task_outcomes"][0]:
                passed = [s["task_id"] for s in a["task_outcomes"] if s.get("passed")]
                failed = [s["task_id"] for s in a["task_outcomes"] if not s.get("passed")]
                out.append("\n**Passed tasks:** " + (", ".join(passed) or "none") + ".\n\n")
                out.append("**Failed tasks:** " + (", ".join(failed) or "none") + ". Full executor outcomes are in the linked artifact and `sample_outcomes.csv`.\n\n")
            else:
                correct = sum(s.get("correct_slots", 0) for s in a["task_outcomes"])
                out.append(f"\n**Recorded per-canvas correct-slot total:** {correct}. Per-slot truth values and worksheet outcomes are retained in `sample_outcomes.csv` and the original result.\n\n")
            out.append("**Related log records:** " + ", ".join(r["ref"] for r in a["related_conversation_records"]) + ".\n\n</details>\n\n")
    out.append("## Appendix C. Exact toys, training audits, diagnostic reports, and cached sweeps\n\n")
    diagnostics = [
        "infogain_lab/toy.json", "infogain_lab/finite_response_audit.json", "infogain_lab/gradient_effect_audit.json",
        "latent_risk_mbpp/validation.json", "latent_risk_mbpp_v2/validation.json", "latent_risk_mbpp_v2/calibration_audit.json",
        "latent_risk_mbpp_v2/body_gate_validation.json", "latent_risk_mbpp_v2/body_gate_token_validation.json", "latent_risk_mbpp_v2/body_gate_scalar_validation.json",
        "latent_value_mbpp/validation.json", "latent_value_mbpp/feature_audit.json", "latent_value_mbpp/provenance.json",
        "context_reconstruction_audit/audit.json", "ugc_diagnostic/diagnostic.json", "latent_count_toys/diagnostic.json",
        "latent_three_term_algebra/verification.json", "latent_allowance_toys/diagnostic.json",
        "latent_gradient_pool100/failure_probe40.json", "latent_full_humaneval164/evaluator_validation164.json",
        "latent_humaneval20/evaluator_validation.json", "latent_pca_flow_humaneval20_lean/diagnostic.json",
        "latent_body_progress_humaneval20/gradient_capture_audit.json", "latent_context_flow_arithmetic/gradient_capture_audit.json",
    ]
    for name in diagnostics:
        path = ROOT / "outputs" / name
        if not path.exists():
            continue
        data = read_json(path)
        out.append(f"### {name}\n\n{link(path, 'Original exact diagnostic artifact')}.\n\n")
        if name.endswith("toy.json"):
            data = {k: v for k, v in data.items() if k != "records"}
        elif name.endswith("gradient_effect_audit.json"):
            data = {"summary": data["summary"], "individual_actions": len(data["records"])}
        elif name == "context_reconstruction_audit/audit.json":
            data = {k: compact(v) for k, v in data.items() if k != "candidates"}
        elif name.endswith("diagnostic.json") and "pca_flow" in name:
            data = compact(data)
        out.append(fence(data) + "\n")
    for name in ("toy_sweep.csv", "arithmetic_seed0_sweep.csv"):
        path = ROOT / "outputs/infogain_lab" / name
        rows = list(csv.DictReader(path.open()))
        out.append(f"### Cached score sweep: {name}\n\nAll {len(rows)} recorded rows, without new model work. {link(path, 'Original CSV')}.\n\n")
        out.append(table(list(rows[0]), [[r[k] for k in rows[0]] for r in rows]))
    out.append("### Earlier-budget correctness replay: all recorded methods and budgets\n\n")
    path = ROOT / "outputs/latent_body_progress_budget_replay/analysis.json"
    data = read_json(path)
    out.append(table(["Method", "Budget", "Passed /20", "Effective steps"], [(s["method"], s["budget"], s["passed"], s.get("effective_steps")) for s in data["studies"]]))
    out.append(data["note"] + "\n\n")
    out.append("### Retrospective body-timing audit: every recorded method\n\n")
    data = read_json(ROOT / "outputs/latent_body_progress_audit/analysis.json")
    out.append(table(["Method", "Passed", "Body ready mean step", "Body multi-token decisions", "Mean body mask holes", "Safe EOS examples", "Mean safe EOS step"], [(s["method"], s["passed"], s["mean_rendered_body_ready_step"], s["body_multi_steps"], s["mean_body_mask_holes"], s["safe_eos_examples"], s["mean_safe_eos_step"]) for s in data["studies"]]))
    out.append(data["note"] + "\n\n")
    out.append("## Appendix D. Recovered experiment, validation, and analysis commands\n\n")
    out.append("The following commands were recorded in the original conversation. Inspection commands are distinguished from launches because reading an old result is not a new run. Failed or interrupted commands remain historical evidence. A related response can contain asynchronous launch output rather than the final score. Full source edits are represented by archived snapshots, rather than duplicated patches.\n\n")
    for c in commands:
        out.append(f"<details>\n<summary>{c['command_id']} — {c['ref']} — {c['time']} — {c['category']}</summary>\n\n")
        out.append(f"Tool response reference: `{c['output_ref']}`.\n\n```bash\n{c['command']}\n```\n\n</details>\n\n")
    out.append("## Appendix E. Public conversation chronology\n\n")
    out.append("Public user requests and assistant progress/results are preserved below, excluding hidden reasoning, system/developer instructions, repeated environment blocks, and automatic goal continuations. Earlier progress statements can be preliminary or superseded; completed artifacts and the status labels above take precedence. Empty messages and the final CLI-resume discussion are retained to make the recovered chronology explicit.\n\n")
    for m in messages:
        body = m["text"].replace("<", "&lt;").replace(">", "&gt;")
        out.append(f"<details>\n<summary>{m['ref']} — {m['time']} — {m['role']}</summary>\n\n{body}\n\n</details>\n\n")
    out.append("## Appendix F. Surviving artifact manifest\n\n")
    out.append("Every experiment source snapshot, JSON result/checkpoint/manifest, CSV, existing report, and run log in the included research directories is listed below. Large binary model/head/geometry tensors remain in their original directories and are linked by study/configuration; they are not copied into this document. The manifest covers textual evidence and records its current SHA-256.\n\n")
    out.append(table(["Family", "Artifact", "Bytes", "SHA-256"], [(a["family"], link(a["path"], a["path"]), a["bytes"], a["sha256"]) for a in artifacts]))
    return "".join(out)


def write_html(text):
    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(text)
    navigation = []
    used = Counter()
    def heading(match):
        level, inner = match.group(1), match.group(2)
        plain = re.sub("<[^>]+>", "", html.unescape(inner))
        slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
        used[slug] += 1
        if used[slug] > 1:
            slug += f"-{used[slug]}"
        if level == "2":
            navigation.append(f'<a href="#{slug}">{html.escape(plain)}</a>')
        return f'<h{level} id="{slug}">{inner}</h{level}>'
    body = re.sub(r"<h([23])>(.*?)</h\1>", heading, body)
    # Keep the HTML copy portable within this checkout and sidecar directory.
    def rewrite(match):
        target = html.unescape(match.group(1))
        if target.startswith(str(ROOT) + "/"):
            target = "../" + target[len(str(ROOT)) + 1:]
        return f'href="{html.escape(target, quote=True)}"'
    body = re.sub(r'href="([^"]+)"', rewrite, body)
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Complete latent scheduler experiments — conversation 01a10259</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;color:#18222e;background:#f5f7fa;font:16px/1.65 system-ui,sans-serif}aside{position:fixed;inset:0 auto 0 0;width:290px;overflow:auto;padding:22px;background:#132333;color:#fff}aside a{display:block;color:#d5e4f4;text-decoration:none;font-size:13px;line-height:1.5;padding:5px 0}main{margin-left:290px;max-width:1450px;padding:40px 48px;background:#fff}h1{font-size:34px;line-height:1.25}h2{font-size:25px;margin-top:54px;border-bottom:2px solid #dae3ee;padding-bottom:12px}h3{font-size:20px;margin-top:32px}a{color:#1462a8;overflow-wrap:anywhere}code{font-size:13px;background:#f1f4f7;padding:2px 4px;overflow-wrap:anywhere}pre{overflow:auto;padding:18px;background:#f1f4f7;border:1px solid #e0e6ed;border-radius:5px;line-height:1.45}pre code{padding:0;background:none}table{display:block;overflow-x:auto;border-collapse:collapse;margin:20px 0;width:100%;font-size:13px}td,th{border:1px solid #dce3eb;padding:8px 10px;text-align:left;vertical-align:top}th{background:#eaf0f7}tr:nth-child(even){background:#f7f9fc}details{margin:12px 0;border:1px solid #dce3eb;border-radius:5px;padding:10px 14px}summary{cursor:pointer;font-weight:600;font-size:14px}details[open]>summary{margin-bottom:14px}.controls{display:flex;gap:10px;flex-wrap:wrap}button{cursor:pointer;border:1px solid #b4c4d8;border-radius:4px;background:#fff;color:#20415f;padding:7px 12px}@media(max-width:1000px){aside{position:static;width:auto;max-height:300px}main{margin:0;padding:25px 20px}}@media print{aside,.controls{display:none}main{margin:0;padding:0;max-width:none}details{break-inside:avoid}table{display:table;font-size:9px}a{color:inherit}}
</style></head><body><aside><strong>Conversation experiment record</strong>"""
    document += "".join(navigation)
    document += '</aside><main><div class="controls"><button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=true)">Expand all evidence</button><button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=false)">Collapse evidence</button><button onclick="window.print()">Print</button></div>'
    document += body + "</main></body></html>"
    REPORT.with_suffix(".html").write_text(document)


def main():
    sources, messages, calls, outputs = load_conversation()
    catalog = build_catalog(calls, outputs)
    commands = parse_commands(calls, outputs)
    roots = {a["study"] for a in catalog} | {"infogain_lab", "latent_risk_mbpp", "latent_risk_mbpp_v2", "latent_value_mbpp", "context_reconstruction_audit", "ugc_diagnostic", "latent_count_toys", "latent_count_ablation_report", "latent_only_ablation", "latent_three_term_algebra", "latent_objective_report", "latent_allowance_toys", "latent_allowance_report", "latent_body_progress_report", "latent_body_progress_audit", "latent_body_progress_budget_replay", "scheduler_scoreboard", "latent_pca_flow_humaneval20"}
    artifacts = scan_auxiliary(roots)
    assert len(catalog) == 573, f"Unexpected artifact count: {len(catalog)}"
    coverage = {
        "conversation_id": CID, "source_streams": sources,
        "scored_arm_records": len(catalog), "study_directories_with_scored_arms": len({a["study"] for a in catalog}),
        "included_textual_artifacts": len(artifacts), "public_messages": len(messages),
        "tool_invocations": len(calls), "paired_tool_responses": len(outputs),
        "recovered_relevant_shell_commands": len(commands),
        "excluded_scored_arms": [a["arm_id"] for a in catalog if a["status"].startswith("EXCLUDED")],
        "scored_arms_without_related_log_reference": [a["artifact"] for a in catalog if not a["related_conversation_records"]],
        "scope": "Existing saved results only; repeated controls, historical failed/invalidated arms retained; no new model experiments.",
    }
    for name, data in (("experiment_catalog.json", catalog), ("commands.json", commands), ("artifact_manifest.json", artifacts), ("coverage.json", coverage)):
        (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    headers = ["arm_id", "family", "study", "method", "seed", "samples", "status", "artifact", "passed", "slot_accuracy", "worksheet_accuracy", "steps", "forward_evaluations_per_sample", "backward_evaluations_per_sample", "partial_backward_evaluations_per_sample", "generation_seconds", "peak_allocated_GiB", "unfinished_masks", "artifact_sha256"]
    with (HERE / "experiment_arms.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for a in catalog:
            row = {k: a.get(k, a["metrics"].get(k)) for k in headers}
            writer.writerow(row)
    with (HERE / "sample_outcomes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm_id", "artifact", "sample_index", "sample_identifier", "outcome_json"])
        for a in catalog:
            for i, s in enumerate(a["task_outcomes"]):
                writer.writerow([a["arm_id"], a["artifact"], i, s.get("task_id", s.get("id", i)), json.dumps(s, ensure_ascii=False)])
    with (HERE / "public_conversation.json").open("w") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    base = REPORT.read_text().split(MARKER)[0].rstrip() + "\n\n" + MARKER + "\n"
    report = base + appendices(sources, messages, catalog, commands, artifacts)
    REPORT.write_text(report)
    write_html(report)
    print(json.dumps(coverage, indent=2))
    print(f"Report: {len(report.split())} whitespace-delimited words; {len(report)} characters")


if __name__ == "__main__":
    main()
