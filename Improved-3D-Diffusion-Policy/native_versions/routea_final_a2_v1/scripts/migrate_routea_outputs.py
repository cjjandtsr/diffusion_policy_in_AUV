#!/usr/bin/env python3
"""Move experiment files from data/outputs into program without changing bytes.
Version 2.1; Python >= 3.8; standard library. Stop all writers; unknown loading assets block apply.
Copies pass SHA256 and byte comparison before deletion; retained weight candidates need Codex review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
VERSION = "2.1"
PROGRAM_ROOT = Path("/home/cjj/AUV/program")
WEIGHT_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors"}
CHECKPOINT_FILES = {"header.json", "state.pt", "SHA256.json", "COMMITTED.json"}
EXPERIMENT_SUFFIXES = {".log", ".txt", ".csv", ".tsv", ".npz", ".npy", ".png",
                       ".jpg", ".jpeg", ".svg", ".pdf", ".mp4", ".tar", ".gz", ".zip", ".jsonl"}
EXPERIMENT_JSON_NAMES = {"metrics.json", "results.json", "report.json", "evaluation.json",
                         "eval.json", "summary.json"}


class MigrationError(RuntimeError):
    pass


def _under(root, relative):
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise MigrationError("path escapes migration root: " + str(path))
    return path


def _roots(repo_root, program_root, audit_dir):
    repo, program, audit = [Path(p).resolve() for p in (repo_root, program_root, audit_dir)]
    source = _under(repo, "data/outputs")
    if not source.is_dir() or program.name != "program":
        raise MigrationError("source must be repository data/outputs; destination must be program")
    if program not in audit.parents:
        raise MigrationError("audit directory must be inside program")
    if source == program or source in program.parents or program in source.parents:
        raise MigrationError("source and program overlap")
    return repo, source, program, audit


def _hash_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _same_bytes(source, target):
    with source.open("rb") as left, target.open("rb") as right:
        while True:
            chunk = left.read(1024 * 1024)
            if chunk != right.read(1024 * 1024):
                return False
            if not chunk:
                return True


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _classification(path, weight_dirs, identities):
    if (path.parent.name.startswith("checkpoint_update_") and path.name in CHECKPOINT_FILES) or path in identities:
        return "retain", "original checkpoint or weight identity/integrity component"
    if re.search(r"normaliz|tokeniz|vocab|anchor|model_config|policy_config", path.name, re.I):
        return "block", "potential loading asset; Codex must determine its use"
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "retain", "weight candidate; model-only acceptance pending Codex review"
    if path.name == "INCOMPLETE_BUDGET":
        return "migrate", "historical training budget evidence; not a model loading asset"
    if (path.parent in weight_dirs and path.suffix.lower() not in EXPERIMENT_SUFFIXES
            and path.name not in EXPERIMENT_JSON_NAMES):
        return "block", "unknown companion beside weights; Codex must determine its use"
    return "migrate", "experiment/evidence file"


def create_plan(repo_root, program_root, audit_dir):
    repo, source, program, audit = _roots(repo_root, program_root, audit_dir)
    paths = sorted(source.rglob("*"))
    files = [p.relative_to(source) for p in paths if p.is_file()]
    weights = [p for p in files if p.suffix.lower() in WEIGHT_SUFFIXES]
    weight_dirs = {p.parent for p in weights}
    identities = {Path(str(p) + suffix) for p in weights
                  for suffix in (".sha256", ".sha256.json", ".identity.json")}
    inventory = []
    for relative in files:
        path = _under(source, relative)
        action, reason = _classification(relative, weight_dirs, identities)
        inventory.append({"relative_path": relative.as_posix(), "action": action, "reason": reason,
                          "size_bytes": path.stat().st_size, "sha256": _hash_file(path)})
    plan = {"schema_version": 2, "tool_version": VERSION, "repo_root": str(repo),
            "source_root": str(source), "program_root": str(program), "audit_dir": str(audit),
            "archive_root": str(audit / "migrated_outputs"), "inventory": inventory,
            "directories": [p.relative_to(source).as_posix() for p in paths if p.is_dir()],
            "blockers": [e["relative_path"] + ": " + e["reason"] for e in inventory if e["action"] == "block"],
            "copy_bytes": sum(e["size_bytes"] for e in inventory if e["action"] == "migrate"),
            "retained_weight_candidates": [p.as_posix() for p in weights if not
                (p.parent.name.startswith("checkpoint_update_") and p.name == "state.pt")],
            "model_only_acceptance": "pending_codex_review"}
    audit.mkdir(parents=True, exist_ok=False)
    _write_json(audit / "inventory.json", {"files": inventory, "directories": plan["directories"]})
    _write_json(audit / "path_mapping.json", {"files": [dict(e,
        source_path=str(_under(source, e["relative_path"])),
        destination_path=str(_under(audit / "migrated_outputs", e["relative_path"])))
        for e in inventory if e["action"] == "migrate"]})
    _write_json(audit / "plan.json", plan)
    return plan


def apply_plan(plan_path, plan_sha256, expected_repo_root, expected_program_root):
    plan_path = Path(plan_path).resolve()
    if _hash_file(plan_path) != plan_sha256:
        raise MigrationError("plan SHA256 differs")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repo, source, program, audit = _roots(expected_repo_root, expected_program_root, plan_path.parent)
    if (plan["tool_version"] != VERSION or plan_path != audit / "plan.json" or
            any(plan[key] != str(value) for key, value in (("repo_root", repo), ("source_root", source),
                ("program_root", program), ("audit_dir", audit), ("archive_root", audit / "migrated_outputs")))):
        raise MigrationError("plan version/paths differ from this tool and its approved roots")
    if plan["blockers"]:
        raise MigrationError("plan has blockers: " + "; ".join(plan["blockers"]))
    if (audit / "apply_summary.json").exists():
        raise MigrationError("apply already attempted; preserve its evidence and return to Codex")
    moves = [e for e in plan["inventory"] if e["action"] == "migrate"]
    summary = {"status": "failed", "plan_sha256": plan_sha256, "deleted_files": 0,
               "source_root": str(source), "archive_root": plan["archive_root"],
               "retained_weight_candidates": plan["retained_weight_candidates"],
               "model_only_acceptance": "pending_codex_review"}
    try:
        for entry in moves:
            original = _under(source, entry["relative_path"])
            target = _under(audit / "migrated_outputs", entry["relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        # Every source and copy is checked before the first deletion.
        for entry in plan["inventory"]:
            original = _under(source, entry["relative_path"])
            if _hash_file(original) != entry["sha256"]:
                raise MigrationError("source changed since plan: " + entry["relative_path"])
            if entry["action"] == "migrate":
                target = _under(audit / "migrated_outputs", entry["relative_path"])
                if _hash_file(target) != entry["sha256"] or not _same_bytes(original, target):
                    raise MigrationError("copy SHA256/bytes differ: " + entry["relative_path"])
        for entry in moves:
            original = _under(source, entry["relative_path"])
            target = _under(audit / "migrated_outputs", entry["relative_path"])
            if any(_hash_file(p) != entry["sha256"] for p in (original, target)):
                raise MigrationError("file changed before deletion: " + entry["relative_path"])
            original.unlink()
            summary["deleted_files"] += 1
        for relative in sorted(plan["directories"], key=lambda p: len(Path(p).parts), reverse=True):
            directory = _under(source, relative)
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        for entry in plan["inventory"]:
            if entry["action"] == "retain" and _hash_file(_under(source, entry["relative_path"])) != entry["sha256"]:
                raise MigrationError("retained weight/component changed: " + entry["relative_path"])
        summary.update(status="complete", retained_files=len(plan["inventory"]) - len(moves),
                       verified_copy_bytes=plan["copy_bytes"], retained_hashes_unchanged=True,
                       all_migrated_targets_sha256_and_bytes_verified_before_delete=True)
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        _write_json(audit / "apply_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, help="new plan directory inside program")
    parser.add_argument("--apply-plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        if args.apply_plan and args.plan_sha256 and not args.audit_dir:
            result = apply_plan(args.apply_plan, args.plan_sha256, repo, PROGRAM_ROOT)
        elif args.audit_dir and not args.apply_plan and not args.plan_sha256:
            plan = create_plan(repo, PROGRAM_ROOT, args.audit_dir)
            result = {"mode": "plan", "plan_path": str(args.audit_dir / "plan.json"),
                      "plan_sha256": _hash_file(args.audit_dir / "plan.json"),
                      "inventory_files": len(plan["inventory"]), "copy_bytes": plan["copy_bytes"],
                      "blockers": plan["blockers"]}
        else:
            raise MigrationError("use --audit-dir OR --apply-plan with --plan-sha256")
        print(json.dumps(result, ensure_ascii=False))
        return 2 if result.get("blockers") else 0
    except (MigrationError, OSError, ValueError, KeyError, TypeError) as exc:
        print("MIGRATION_BLOCKED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
