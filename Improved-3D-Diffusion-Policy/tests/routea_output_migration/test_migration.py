"""Byte preservation and failure contracts for the outputs migration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/migrate_routea_outputs.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MigrationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("output_migration", SCRIPT)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.source = self.repo / "data/outputs"
        self.source.mkdir(parents=True)
        self.program = Path(self.temp.name) / "program"
        self.audit = self.program / "migration_v2"

    def put(self, name, content=b"original evidence\x00\xff\n"):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def apply(self):
        plan_path = self.audit / "plan.json"
        return self.api.apply_plan(plan_path, digest(plan_path), self.repo, self.program)

    def test_weight_and_checkpoint_bytes_stay_while_evidence_moves_unchanged(self):
        checkpoint = "r1/seed1/checkpoint_update_000032/"
        header = self.put(checkpoint + "header.json", b'{"schema_version":2}\n')
        state = self.put(checkpoint + "state.pt", b"original trained weight")
        sums = {"header.json": digest(header), "state.pt": digest(state)}
        self.put(checkpoint + "SHA256.json", json.dumps(sums).encode())
        self.put(checkpoint + "COMMITTED.json", json.dumps({
            "schema_version": 2, "status": "committed", "files": sums,
            "header_schema_hash": sums["header.json"]}).encode())
        retained = {checkpoint + name for name in self.api.CHECKPOINT_FILES}
        for suffix in (".ckpt", ".pt", ".pth", ".safetensors"):
            name = "legacy/model" + suffix
            self.put(name, b"uninspected weight candidate")
            retained.add(name)
        self.put("legacy/model.ckpt.sha256", b"original identity\n")
        retained.add("legacy/model.ckpt.sha256")
        moved = {checkpoint + "plots/trace.npy", "run/metadata.json", "run/config.yaml"}
        for name in moved:
            self.put(name)
        original = {p.relative_to(self.source).as_posix(): p.read_bytes()
                    for p in self.source.rglob("*") if p.is_file()}
        plan = self.api.create_plan(self.repo, self.program, self.audit)
        self.assertEqual(plan["blockers"], [])
        self.assertEqual(original, {p.relative_to(self.source).as_posix(): p.read_bytes()
                                    for p in self.source.rglob("*") if p.is_file()})
        result = self.apply()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["deleted_files"], len(moved))
        self.assertEqual(result["model_only_acceptance"], "pending_codex_review")
        for name in retained:
            self.assertEqual((self.source / name).read_bytes(), original[name])
        for name in moved:
            self.assertFalse((self.source / name).exists())
            self.assertEqual((self.audit / "migrated_outputs" / name).read_bytes(), original[name])
        mapping = json.loads((self.audit / "path_mapping.json").read_text(encoding="utf-8"))
        self.assertEqual({e["relative_path"] for e in mapping["files"]}, moved)
        self.assertEqual({p.name for p in header.parent.iterdir()}, self.api.CHECKPOINT_FILES)

    def test_late_copy_failure_keeps_every_source_and_records_failure(self):
        paths = [self.put("run/a.log"), self.put("run/b.log")]
        self.api.create_plan(self.repo, self.program, self.audit)
        copy = self.api.shutil.copyfileobj
        calls = 0

        def interrupted(source, target, length):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("copy interrupted")
            return copy(source, target, length)

        with mock.patch.object(self.api.shutil, "copyfileobj", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "copy interrupted"):
                self.apply()
        self.assertTrue(all(path.exists() for path in paths))
        self.assertTrue((self.audit / "apply_summary.json").is_file())
        summary = json.loads((self.audit / "apply_summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["status"], summary["deleted_files"]), ("failed", 0))

    def test_incomplete_budget_evidence_moves_but_model_checkpoint_stays(self):
        directory = "g8_config_anchor_25ae259/"
        weight = self.put(directory + "dppo_uav_envstep_000001600_ppoupd_000001.pt",
                          b"historical model recovery checkpoint")
        marker = self.put(directory + "INCOMPLETE_BUDGET", b'{"cumulative_transitions":1600}\n')
        metrics = self.put(directory + "metrics.jsonl")
        originals = {path.name: path.read_bytes() for path in (weight, marker, metrics)}
        plan = self.api.create_plan(self.repo, self.program, self.audit)
        self.assertEqual(plan["blockers"], [])
        self.assertEqual(self.apply()["deleted_files"], 2)
        self.assertEqual(weight.read_bytes(), originals[weight.name])
        for evidence in (marker, metrics):
            self.assertFalse(evidence.exists())
            target = self.audit / "migrated_outputs" / directory / evidence.name
            self.assertEqual(target.read_bytes(), originals[evidence.name])

    def test_copy_corruption_keeps_every_source(self):
        paths = [self.put("run/a.log"), self.put("run/b.log")]
        self.api.create_plan(self.repo, self.program, self.audit)
        copy = self.api.shutil.copyfileobj
        calls = 0

        def corrupt(source, target, length):
            nonlocal calls
            calls += 1
            if calls == 2:
                return target.write(b"corrupted copy")
            return copy(source, target, length)

        with mock.patch.object(self.api.shutil, "copyfileobj", side_effect=corrupt):
            with self.assertRaises(self.api.MigrationError):
                self.apply()
        self.assertTrue(all(path.exists() for path in paths))
        self.assertTrue((self.audit / "apply_summary.json").is_file())

    def test_unknown_loading_assets_block_apply(self):
        self.put("legacy/model.ckpt")
        unknown = {"legacy/config.yaml", "assets/normalizer.pt", "assets/anchor_bank.npz"}
        for name in unknown:
            self.put(name)
        log = self.put("run/eval.log")
        plan = self.api.create_plan(self.repo, self.program, self.audit)
        self.assertEqual({e["relative_path"] for e in plan["inventory"]
                          if e["action"] == "block"}, unknown)
        with self.assertRaisesRegex(self.api.MigrationError, "blockers"):
            self.apply()
        self.assertTrue(log.exists())


if __name__ == "__main__":
    unittest.main()
