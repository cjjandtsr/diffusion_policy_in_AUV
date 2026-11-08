"""Run the unchanged G3 worker with environment-matching goal validation precision."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys


def install_goal_precision(episodes):
    before = inspect.getsource(episodes.validate_rows)
    old = 'goal = np.asarray(header["task"]["goal_pos"], np.float64)'
    new = 'goal = np.asarray(header["task"]["goal_pos"], np.float32)'
    if before.count(old) != 1:
        raise ValueError('unexpected native validator; expected the original float64 goal line')
    after = before.replace(old, new)
    namespace = dict(episodes.__dict__)
    exec(compile(after, '<routea-goal-float32-validator>', 'exec'), namespace)
    episodes.validate_rows = namespace['validate_rows']
    return dict(scope='evaluation validation only; native files unchanged',
                native_module=str(Path(episodes.__file__).resolve()),
                native_file_sha256=hashlib.sha256(Path(episodes.__file__).read_bytes()).hexdigest(),
                original_function_sha256=hashlib.sha256(before.encode()).hexdigest(),
                corrected_function_sha256=hashlib.sha256(after.encode()).hexdigest(),
                change=dict(old=old, new=new), tolerance_changed=False)


def main():
    path = Path(__file__).with_name('run_routea_xk_evaluation.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != '15bca3863d426f881236c2dcca76bcab7cc5fa6ed3d9011e27ebf8f05d3d39e7':
        raise ValueError('requires the unchanged G3 worker')
    spec = importlib.util.spec_from_file_location('routea_g3_unchanged_worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    native = worker.native_system

    def load(args):
        result = native(args)
        from scripts.routea_paired_support import episodes
        Path(episodes.__file__).resolve().relative_to(Path(args.source_root).resolve())
        report = install_goal_precision(episodes)
        report['compatibility_entry_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'validation_precision.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return result

    worker.native_system = load
    if len(sys.argv) < 2 or sys.argv[1] != 'worker':
        raise ValueError('compatibility entry is restricted to worker evaluation')
    worker.main()


if __name__ == '__main__':
    main()
