"""Make the staged rebuild importable beside the installed WSL project.

The delivery tree intentionally does not contain replacement ``__init__.py``
files for ``diffusion_policy_3d`` or ``diffusion_policy_3d.rl``.  During staging
verification, however, WSL loads the existing project's regular ``rl`` package
first.  Extending that package's search path lets pytest exercise the staged
subtree without modifying or shadowing the user's existing package files.
"""

from pathlib import Path

import diffusion_policy_3d.rl as rl_package


STAGED_PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGED_RL_PATH = STAGED_PROJECT_ROOT / "diffusion_policy_3d" / "rl"

staged_rl = str(STAGED_RL_PATH)
if staged_rl not in rl_package.__path__:
    rl_package.__path__.insert(0, staged_rl)
