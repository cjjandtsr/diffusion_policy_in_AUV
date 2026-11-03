"""Evaluate the fixed val20 bank with the paper's open-space v3 protocol.

The checkpoint and its training config remain unchanged. Only the external
evaluation adapter disables the legacy artificial space termination, as in
the paper's P2/P3 closed-loop evaluation.
"""
from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
from scripts import run_routea_val20_endpoint as val20


def install_open_space_adapter():
    original = endpoint._build_adapter

    def build_adapter(*args, **kwargs):
        adapter = original(*args, **kwargs)
        adapter.environment.terminate_at_space_limits = False
        assert adapter.environment.space_cost_budget is None
        print('P4B_EVAL_SPACE_TERMINATION_DISABLED', flush=True)
        return adapter

    endpoint._build_adapter = build_adapter


def main(argv=None):
    install_open_space_adapter()
    return val20.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
