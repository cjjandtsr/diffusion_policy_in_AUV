#!/usr/bin/env python
"""P2 probe: official DPPO vendor API extraction.

Two independent checks:
  A) Static AST extraction (works even if imports fail)
  B) Runtime import + inspect.signature (resolves real classes)
"""
import ast
import inspect
import os
import sys
import traceback

VENDOR = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/third_party/dppo_official_cc7234a"

FILES = [
    "model/diffusion/diffusion.py",
    "model/diffusion/diffusion_vpg.py",
    "model/diffusion/diffusion_ppo.py",
    "model/diffusion/sampling.py",
    "model/common/critic.py",
]

TARGETS = {
    "DiffusionModel": ["p_mean_var", "forward", "q_sample", "p_losses", "loss"],
    "VPGDiffusion": [
        "__init__", "p_mean_var", "forward", "get_logprobs",
        "get_logprobs_subsample", "loss", "step",
    ],
    "PPODiffusion": ["__init__", "loss"],
    "CriticObs": ["__init__", "forward"],
}

def expr_name(node):
    """Minimal Name/Attribute walker (ast.unparse unavailable on py3.8)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{expr_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{expr_name(node.value)}[{expr_name(node.slice)}]"
    return "?"


def args_str(args):
    parts = [a.arg for a in args.posonlyargs]
    parts += [a.arg for a in args.args]
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    parts += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return ", ".join(parts)


print("=" * 80)
print("PART A: STATIC AST EXTRACTION")
print("=" * 80)
for rel in FILES:
    path = os.path.join(VENDOR, rel)
    try:
        src = open(path).read()
    except OSError as e:
        print(f"[AST] FAIL to read {rel}: {e}")
        continue
    tree = ast.parse(src)
    print(f"\n--- {rel} ---")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(expr_name(b) for b in node.bases)
            print(f"class {node.name}({bases})")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    ret = (
                        f" -> {expr_name(item.returns)}"
                        if item.returns is not None else ""
                    )
                    print(f"    def {item.name}({args_str(item.args)}){ret}")
        elif isinstance(node, ast.FunctionDef):
            print(f"def {node.name}({args_str(node.args)})")

print()
print("=" * 80)
print("PART B: RUNTIME IMPORT + inspect.signature")
print("=" * 80)
sys.path.insert(0, VENDOR)
try:
    from model.diffusion.diffusion import DiffusionModel, Sample
    from model.diffusion.diffusion_vpg import VPGDiffusion
    from model.diffusion.diffusion_ppo import PPODiffusion
    from model.diffusion import sampling
    from model.common.critic import CriticObs, CriticObsAct
    IMPORT_OK = True
except Exception:
    IMPORT_OK = False
    print("IMPORT FAILED; full traceback:")
    traceback.print_exc()

if IMPORT_OK:
    print("IMPORT_OK")
    for cls in [DiffusionModel, VPGDiffusion, PPODiffusion, CriticObs, CriticObsAct]:
        print(f"\n### {cls.__name__}")
        print(f"file: {inspect.getfile(cls)}")
        mro = [c.__name__ for c in cls.__mro__]
        print(f"MRO: {mro}")
        try:
            print(f"__init__{inspect.signature(cls.__init__)}")
        except (TypeError, ValueError) as e:
            print(f"__init__ signature unavailable: {e}")
        for meth in TARGETS.get(cls.__name__, []):
            m = getattr(cls, meth, None)
            if m is None:
                print(f"  [MISSING] {meth}")
                continue
            try:
                print(f"  {meth}{inspect.signature(m)}")
            except (TypeError, ValueError) as e:
                print(f"  {meth}: signature unavailable ({e})")

    print("\n### Sample namedtuple fields:")
    print(f"Sample._fields = {Sample._fields}")

    print("\n### diffusion.py module-level defs:")
    for name, obj in inspect.getmembers(sampling):
        if inspect.isfunction(obj):
            try:
                print(f"  {name}{inspect.signature(obj)}  file={inspect.getfile(obj)}")
            except (TypeError, ValueError):
                print(f"  {name}: signature unavailable")

    # verify resolved files are inside vendor root
    print("\n### resolved-file-inside-vendor check:")
    for cls in [DiffusionModel, VPGDiffusion, PPODiffusion, CriticObs]:
        f = inspect.getfile(cls)
        inside = os.path.realpath(f).startswith(os.path.realpath(VENDOR))
        print(f"  {cls.__name__}: inside vendor = {inside} ; path = {f}")

    # conflict check: what would 'import model' resolve to without explicit sys.path?
    import model
    print(f"\n### bare 'import model' resolves to: {model.__file__}")
