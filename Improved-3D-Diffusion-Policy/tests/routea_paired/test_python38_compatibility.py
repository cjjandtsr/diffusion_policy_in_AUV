"""Guard known Python >3.8 standard-library calls in the delivered evaluator."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class Python38Compatibility(unittest.TestCase):
    def test_delivered_python_uses_python38_syntax_and_known_api_baseline(self):
        paths=[ROOT/"scripts/run_routea_paired_comparison.py"]
        paths+=sorted((ROOT/"scripts/routea_paired_support").glob("*.py"))
        paths+=sorted((ROOT/"tests/routea_paired").glob("*.py"))
        issues=[]
        newer_methods={"removeprefix","removesuffix","is_relative_to","with_stem",
                       "readlink","hardlink_to","bit_count"}
        for p in paths:
            source=p.read_text(encoding="utf-8")
            try:
                tree=ast.parse(source,filename=str(p),feature_version=8)
            except SyntaxError as exc:
                issues.append(str(p.relative_to(ROOT))+": syntax: "+str(exc))
                continue
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    method=node.func.attr if isinstance(node.func,ast.Attribute) else None
                    if method in newer_methods:
                        issues.append("%s:%d Python>3.8 API %s"%(p.relative_to(ROOT),node.lineno,method))
                    if method=="write_text" and any(k.arg=="newline" for k in node.keywords):
                        issues.append("%s:%d Path.write_text newline needs Python3.10"%(p.relative_to(ROOT),node.lineno))
                    if (isinstance(node.func,ast.Name) and node.func.id=="zip"
                            and any(k.arg=="strict" for k in node.keywords)):
                        issues.append("%s:%d zip strict needs Python3.10"%(p.relative_to(ROOT),node.lineno))
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ""]
                    if any(m.split(".")[0] in ("zoneinfo","graphlib","tomllib") for m in modules):
                        issues.append("%s:%d newer standard-library module"%(p.relative_to(ROOT),node.lineno))
        self.assertEqual(issues,[],"\n".join(issues))


if __name__=="__main__":
    unittest.main()
