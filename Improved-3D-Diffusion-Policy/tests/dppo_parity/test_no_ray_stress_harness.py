"""R2 rework P1-4: no-ray stress harness is versioned, self-consistent.

The audit finding: the R2.5 stress ran 9,000 wrapper-level frames + 1,000
adapter steps (not 10,000 ROLLOUT frames) and the harness lived in temp/
(uncommitted). These tests pin the VERSIONED harness
(scripts/no_ray_stress.py):

- every counted frame goes through the real adapter path (no wrapper-level
  loop in the harness anymore);
- the 10,000-frame minimum is a HARD pass condition: a 30-frame run must
  FAIL (negative pin — a harness that passed short runs would violate the
  guide Step 4 contract);
- the evidence JSON binds command / exit code / tree hash (P2-10).
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "scripts" / "no_ray_stress.py"


def _run_harness(tmp_path: Path, frames: int, seed: int, tag: str) -> dict:
    out = tmp_path / f"{tag}.json"
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--frames", str(frames),
         "--seed", str(seed), "--output", str(out)],
        capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT),
    )
    assert proc.returncode in (0, 1), f"harness crashed:\n{proc.stderr}"
    return proc.returncode, json.loads(out.read_text())


def test_no_ray_stress_harness_smoke(tmp_path):
    """30 rollout frames: correct accounting + the 10k gate refuses PASS."""
    rc, ev = _run_harness(tmp_path, frames=30, seed=0, tag="smoke")
    # invalid=0 and far-dome evidence are present even on a short run
    assert ev["total_frames"] == 30
    assert ev["invalid"] == 0
    assert ev["stats"]["invalid_reasons"] == []
    assert ev["far_dome_frames"] > 0, "degenerate band frames must occur"
    assert ev["report_failures"] == 0
    assert ev["stats"]["resets"] > 0
    # the harness has no wrapper-level loop: every frame is an adapter step
    # P2-10 evidence binding
    assert "no_ray_stress.py" in ev["command"]
    assert len(ev["tree_hash"]) == 40
    # 10,000-frame minimum is a HARD gate: a short run must NOT pass
    assert ev["pass"] is False
    assert rc == 1
    assert ev["exit_code"] == 1


def test_no_ray_stress_evidence_binds_seed_and_stats(tmp_path):
    """Evidence records the exact seed and the per-frame accounting."""
    rc, ev = _run_harness(tmp_path, frames=30, seed=42, tag="seed42")
    assert ev["seed"] == 42
    assert ev["stats"]["n_frames"] == 30
    assert ev["stats"]["n_frames"] == ev["total_frames"]
    assert ev["stats"]["invalid_reasons"] == []
    assert rc == 1  # short run: gate refuses
