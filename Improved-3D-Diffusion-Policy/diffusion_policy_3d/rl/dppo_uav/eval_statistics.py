"""R6-D (R6-EVAL-1): paired-eval statistics layer (F-04 / BC pseudo-
replication).

Guide §11.5.1/§11.5.2/§11.5.5 + §7.1 — the statistics layer is PURE
aggregation, it never rolls out:

- PairUnitKey: the four-field identity (bank_id, episode_config_hash,
  replica_id, policy_checkpoint_hash). pair_unit_key derives it from
  artifact provenance + record and hard-fails on missing hash / empty
  provenance / RNG provenance mismatch (R5.4 re-derivation) — never a
  silent default;
- BC deduplication: identical BC records across RL artifacts count ONCE
  (the 18/24 counterexample — raw=24, unique/effective=12, SR=9/12);
  different replicas are kept; different RL checkpoints form policy
  clusters, NOT independent policies (3 seeds != 180 independent);
- binary-outcome aggregation: pooled SR with a Wilson CI fixed oracle,
  entry-level cluster SR (unweighted per-episode_config_hash mean over
  unique records) and the report ALWAYS carrying raw n / unique n /
  cluster n — never a single n;
- paired_policy_delta: per-episode paired deltas over common episode
  keys; provenance mismatch or unpaired episodes hard-fail; identical
  policy + common keys => every delta exactly 0;
- validate_paired_matrix: a 12x5 matrix missing one replica is
  incomplete; duplicate episode keys hard-fail.

The records are duck-typed over the EpisodePairedRecord surface
(episode_config["episode_config_hash"], replica_id, policy_noise_seed,
rng_contract_version, bc/rl outcome dicts, episode_config content).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from diffusion_policy_3d.rl.dppo_uav.paired_rng import policy_noise_seed

# Side names accepted by aggregation (which outcome dict a section pools).
SIDES = ("bc", "rl")

# Wilson score interval z value (fixed oracle, tests pin the closed form).
WILSON_Z = 1.96


@dataclass(frozen=True)
class PairUnitKey:
    """The four-field statistical unit identity (§11.5.1).

    (bank_id, episode_config_hash, replica_id, policy_checkpoint_hash)
    — the same BC checkpoint appearing in three RL artifacts deduplicates
    to ONE unit per (episode, replica); a different RL checkpoint is a
    DIFFERENT policy identity (policy seed), never silently merged.
    """

    bank_id: str
    episode_config_hash: str
    replica_id: int
    policy_checkpoint_hash: str


def pair_unit_key(
    bank_id: str, checkpoint_hash: str, record: Any,
) -> PairUnitKey:
    """Derive the unit key from artifact provenance + record.

    The checkpoint hash comes from the ARTIFACT (policy provenance), the
    episode identity from the record's frozen episode config hash, the
    replica from the record. Hard-fails — never a silent default:

    - empty bank_id / empty checkpoint_hash -> "bank_id"/"checkpoint_hash";
    - missing/non-str episode_config_hash -> "episode_config_hash";
    - the record's policy_noise_seed must re-derive from
      (bank_id, episode_config_hash, replica_id) via the R5.4 formula —
      a tampered seed breaks the paired-RNG contract -> "RNG provenance".
    """
    if not isinstance(bank_id, str) or not bank_id:
        raise ValueError(
            "pair unit key requires a non-empty bank_id — the bank the "
            "episode belongs to is part of the unit identity"
        )
    if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
        raise ValueError(
            "pair unit key requires a non-empty policy checkpoint_hash — "
            "the policy identity is part of the unit identity"
        )
    ep_config = record.episode_config
    ep_hash = (
        ep_config.get("episode_config_hash")
        if isinstance(ep_config, dict) else None
    )
    if not isinstance(ep_hash, str) or not ep_hash:
        raise ValueError(
            "pair unit key requires a non-empty episode_config_hash in "
            "the record's episode_config"
        )
    replica = record.replica_id
    if isinstance(replica, bool) or not isinstance(replica, int) or replica < 0:
        raise ValueError(
            "pair unit key requires replica_id to be an int >= 0, got "
            f"{replica!r}"
        )
    expected = policy_noise_seed(bank_id, ep_hash, replica)
    if record.policy_noise_seed != expected:
        raise ValueError(
            "RNG provenance mismatch: policy_noise_seed "
            f"{record.policy_noise_seed} != re-derived {expected} from "
            "(bank_id, episode_config_hash, replica_id) — the recorded "
            "noise stream is not reproducible from the artifact "
            "provenance"
        )
    return PairUnitKey(
        bank_id=bank_id,
        episode_config_hash=ep_hash,
        replica_id=replica,
        policy_checkpoint_hash=checkpoint_hash,
    )


def pair_unit_key_from_artifact(artifact: Any, record: Any) -> PairUnitKey:
    """pair_unit_key from an artifact-like object (EvalArtifact surface:
    bank_id + checkpoint_hash)."""
    return pair_unit_key(
        getattr(artifact, "bank_id", None),
        getattr(artifact, "checkpoint_hash", None),
        record,
    )


@dataclass(frozen=True)
class DeduplicatedRecords:
    """The dedup result (§11.5.2).

    raw_records = what the artifacts hold; unique_records = the distinct
    statistical units after identical (episode, replica, policy) records
    count once; effective_n = the n statistics use (== unique_records);
    policy_clusters = the number of distinct policy checkpoints in the
    set (3 RL seeds are 3 clusters, never 180 independent policies);
    episodes = the unique records themselves, for downstream aggregation.
    """

    raw_records: int
    unique_records: int
    effective_n: int
    policy_clusters: int
    episodes: Tuple[Any, ...]


def _record_episode_hash(record: Any) -> str:
    """The record's frozen episode_config_hash — required, never empty."""
    ep_config = record.episode_config
    ep_hash = (
        ep_config.get("episode_config_hash")
        if isinstance(ep_config, dict) else None
    )
    if not isinstance(ep_hash, str) or not ep_hash:
        raise ValueError(
            "record without a non-empty episode_config_hash cannot be "
            "deduplicated or paired — the frozen episode identity is "
            "required"
        )
    return ep_hash


def deduplicate_policy_records(
    records: Any,
    bank_id: Optional[str] = None,
    policy_checkpoint_hash: Optional[str] = None,
    bc_checkpoint_hash: Optional[str] = None,
) -> DeduplicatedRecords:
    """The 18/24 counterexample fix (§11.5.2).

    Records are distinct statistical units iff their
    (episode_config_hash, replica_id, policy_checkpoint_hash) triple
    differs. The same BC checkpoint appearing in several RL artifacts
    therefore counts ONCE per (episode, replica) — the effective
    baseline is 9/12 on 12 tasks, NOT 18/24 on 24. Different replicas
    are kept (each is its own noise draw); different policy checkpoints
    form clusters, not independence.

    The checkpoint hash is read from each record's policy_checkpoint_hash
    (artifact provenance mirrored onto the record), or supplied once via
    policy_checkpoint_hash when the caller has it artifact-level. When a
    bank_id is given, every record's policy_noise_seed is re-derived and
    must match (R5.4) — tampered provenance hard-fails.
    """
    if bank_id is not None and (not isinstance(bank_id, str) or not bank_id):
        raise ValueError(
            "deduplicate_policy_records requires a non-empty bank_id"
        )
    if (policy_checkpoint_hash is not None
            and (not isinstance(policy_checkpoint_hash, str)
                 or not policy_checkpoint_hash)):
        raise ValueError(
            "deduplicate_policy_records requires a non-empty "
            "policy_checkpoint_hash"
        )
    # R6-EVAL-1 (§11.5.1): BC-side identity normalization — the BC
    # reference is ONE checkpoint shared across every RL artifact (each
    # artifact's episodes carry the RL hash mirrored onto the record), so
    # the BC section deduplicates by the artifact-level bc_checkpoint_hash
    # ("无歧义重算" — the paired record's BC identity re-derives from the
    # artifact), never by the RL hash that happens to sit on the record.
    if (bc_checkpoint_hash is not None
            and (not isinstance(bc_checkpoint_hash, str)
                 or not bc_checkpoint_hash)):
        raise ValueError(
            "deduplicate_policy_records requires a non-empty "
            "bc_checkpoint_hash"
        )
    seen = set()
    unique: list = []
    clusters = set()
    first_by_key: Dict[Any, Any] = {}
    for record in records:
        ep_hash = _record_episode_hash(record)
        ckpt = record.policy_checkpoint_hash or policy_checkpoint_hash or ""
        if not ckpt:
            raise ValueError(
                "record without a policy_checkpoint_hash — dedup needs "
                "the policy identity (mirror the artifact-level "
                "checkpoint_hash onto the record, or pass "
                "policy_checkpoint_hash explicitly)"
            )
        if bank_id is not None:
            expected = policy_noise_seed(bank_id, ep_hash, record.replica_id)
            if record.policy_noise_seed != expected:
                raise ValueError(
                    "RNG provenance mismatch: policy_noise_seed "
                    f"{record.policy_noise_seed} != re-derived {expected} "
                    "from (bank_id, episode_config_hash, replica_id)"
                )
        # The BC section's unit identity is the shared BC reference —
        # bc_checkpoint_hash overrides whatever the record carries, so
        # BC records from every RL artifact collapse onto ONE unit per
        # (episode, replica) (§11.5.1 18/24 counterexample fix).
        unit_ckpt = bc_checkpoint_hash or ckpt
        key = (ep_hash, record.replica_id, unit_ckpt)
        clusters.add(unit_ckpt)
        if key not in seen:
            seen.add(key)
            first_by_key[key] = record
            unique.append(record)
        else:
            # R6-EVAL-1 (P2, §11.5.2): records folded into the SAME
            # statistical unit must AGREE on the episode's evidence —
            # the same (episode, replica, policy) running with
            # conflicting episode config or outcomes in two artifacts is
            # a contradiction, never a coin-flip. Exactly-identical
            # duplicates (e.g. the same BC section read twice) dedup
            # silently.
            first = first_by_key[key]
            if (record.episode_config != first.episode_config
                    or record.bc != first.bc or record.rl != first.rl):
                raise ValueError(
                    "duplicate statistical unit with conflicting "
                    f"content (episode_config_hash={ep_hash[:12]}, "
                    f"replica_id={record.replica_id}, "
                    f"policy_checkpoint_hash={ckpt[:12]}): the same "
                    "episode must report identical evidence in every "
                    "artifact it appears in — a same-unit disagreement "
                    "is a corrupt provenance, not a statistic"
                )
    return DeduplicatedRecords(
        raw_records=len(records),
        unique_records=len(unique),
        effective_n=len(unique),
        policy_clusters=len(clusters),
        episodes=tuple(unique),
    )


def wilson_interval(successes: int, n: int) -> Tuple[float, float]:
    """Wilson score interval (z = WILSON_Z) over the UNIQUE/effective n.

    The closed form is the fixed oracle the tests pin (n=12, SR=9/12 ->
    (0.4677, 0.9111); n=1, SR=1 -> upper EXACTLY 1.0). Sample-internal
    description only (§11.5.2) — it never substitutes for the cluster
    structure.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("wilson_interval requires n >= 1, got {n!r}")
    if (isinstance(successes, bool) or not isinstance(successes, int)
            or not 0 <= successes <= n):
        raise ValueError(
            "wilson_interval requires 0 <= successes <= n, got "
            f"({successes}, {n})"
        )
    z = WILSON_Z
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def aggregate_binary_outcomes(
    records: Any,
    bank_id: Optional[str] = None,
    policy_checkpoint_hash: Optional[str] = None,
    side: str = "rl",
    bc_checkpoint_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Pooled + cluster-aware binary outcome aggregation (§11.5.2).

    Returns raw_n / unique_n / effective_n / cluster_n / policy_clusters
    / success_count / sr / wilson_ci / cluster_sr / collision_count / cr
    / raw_events / terminal_category_distribution:

    - sr pools over the UNIQUE records (the dedup n), never the raw n;
    - cluster_sr is the unweighted per-episode_config_hash mean of the
      per-entry SR over unique records — it can diverge from the pooled
      SR when the matrix is incomplete (entry A 2/2, entry B 1/3 ->
      pooled 3/5 vs cluster 2/3);
    - wilson_ci is the sample-internal interval over the effective n;
    - cr is the raw-event collision rate over the unique records;
    - raw_events (R6-E §11.6.4, F-10) counts the raw Gymnasium
      terminated/truncated flags and the timeout event — timeout counts
      ONLY done_reason == "timeout" (a max_ep_len cut is a rollout_cut,
      never a timeout); a record without the §11.6.4 keys is a
      pre-R6-E artifact and hard-fails — never a fabricated category;
    - terminal_category_distribution is the mutually exclusive category
      count over the unique records — reached and collision both stay
      visible, never collapsed into a priority-silent label;
    - side selects which outcome dict is pooled ("bc" for the BC
      reference section, "rl" for policy sections);
    - bc_checkpoint_hash (R6-EVAL-1, §11.5.1) normalizes the BC side's
      unit identity to the shared BC reference when side="bc" — records
      from every RL artifact then collapse onto one unit per (episode,
      replica); ignored for side="rl".

    Empty input is a hard fail — never an empty dict or a fabricated 0.
    """
    if not records:
        raise ValueError("no records to aggregate — an empty set is not a "
                         "statistic")
    if side not in SIDES:
        raise ValueError(f"aggregate side must be one of {SIDES}, got {side!r}")
    # R6-EVAL-1 (§11.5.1): the BC side's unit identity is the shared BC
    # reference — the artifact-level bc_checkpoint_hash normalizes it
    # across every RL artifact; the RL side clusters by the record's own
    # policy_checkpoint_hash.
    dedup = deduplicate_policy_records(
        records,
        bank_id=bank_id,
        policy_checkpoint_hash=policy_checkpoint_hash,
        bc_checkpoint_hash=(bc_checkpoint_hash if side == "bc" else None),
    )
    outcome = lambda rec: rec.bc if side == "bc" else rec.rl  # noqa: E731
    successes = sum(
        1 for rec in dedup.episodes if bool(outcome(rec)["success"])
    )
    collisions = sum(
        1 for rec in dedup.episodes if bool(outcome(rec)["collision"])
    )
    # R6-E §11.6.4 (F-10): raw event counts + the mutually exclusive
    # terminal-category distribution over the UNIQUE records (the same n
    # the SR pools over). timeout counts only done_reason == "timeout"
    # (the record's own "timeout" flag) — a loop-level cut is rollout_cut,
    # never a timeout.
    raw_events: Dict[str, int] = {"terminated": 0, "truncated": 0,
                                  "timeout": 0}
    category_counts: Dict[str, int] = {}
    for rec in dedup.episodes:
        o = outcome(rec)
        missing = [k for k in ("terminated", "truncated",
                               "terminal_category") if k not in o]
        if missing:
            raise ValueError(
                f"record lacks R6-E §11.6.4 keys {missing} — raw-event/"
                "terminal-category aggregation requires a post-R6-E "
                "artifact"
            )
        raw_events["terminated"] += int(bool(o["terminated"]))
        raw_events["truncated"] += int(bool(o["truncated"]))
        raw_events["timeout"] += int(bool(o.get("timeout", False)))
        category_counts[o["terminal_category"]] = (
            category_counts.get(o["terminal_category"], 0) + 1
        )
    per_entry: Dict[str, list] = {}
    for rec in dedup.episodes:
        per_entry.setdefault(
            _record_episode_hash(rec), []
        ).append(bool(outcome(rec)["success"]))
    entry_srs = [
        sum(flags) / len(flags) for flags in per_entry.values()
    ]
    cluster_sr = float(sum(entry_srs) / len(entry_srs))
    return {
        "raw_n": dedup.raw_records,
        "unique_n": dedup.unique_records,
        "effective_n": dedup.effective_n,
        "cluster_n": len(per_entry),
        "policy_clusters": dedup.policy_clusters,
        "success_count": successes,
        "sr": successes / dedup.unique_records,
        "wilson_ci": wilson_interval(successes, dedup.unique_records),
        "cluster_sr": cluster_sr,
        # R6-E §11.6.4 (F-10): raw-event SR/CR + terminal-category
        # distribution — reached+collision both visible, timeout only
        # when done_reason == "timeout".
        "collision_count": collisions,
        "cr": collisions / dedup.unique_records,
        "raw_events": dict(raw_events),
        "terminal_category_distribution": dict(
            sorted(category_counts.items())),
    }


@dataclass(frozen=True)
class PairedAggregate:
    """Per-episode paired deltas over the common key set (§11.5.2).

    per_key is keyed by (episode_config_hash, replica_id) — the paired
    comparison unit; success_deltas / return_deltas are the per-key
    deltas in the same key order (rl minus bc); mean_success_delta is
    the unweighted mean; all_zero is True iff every success delta is
    exactly 0 (the identical-policy sentinel — binary outcomes need no
    tolerance).
    """

    common_n: int
    unpaired_bc: int
    unpaired_rl: int
    per_key: Dict[Tuple[str, int], int]
    success_deltas: Tuple[int, ...]
    return_deltas: Tuple[float, ...]
    mean_success_delta: float
    all_zero: bool


def paired_policy_delta(bc_records: Any, rl_records: Any) -> PairedAggregate:
    """Per-episode paired BC vs RL deltas (§11.5.2).

    Pairs by (episode_config_hash, replica_id) — the SAME frozen episode
    and SAME policy-noise draw on both sides. Hard-fails:

    - any key on one side without its partner -> "unpaired";
    - rng_contract_version or policy_noise_seed differing between the
      paired records -> "RNG provenance" (the two sides must have drawn
      the identical noise stream);
    - episode_config CONTENT differing between the paired records ->
      "episode config" (same hash, different frozen episode).

    Identical policy over the common keys => every delta exactly 0.
    """
    bc_by = {(_record_episode_hash(r), r.replica_id): r for r in bc_records}
    rl_by = {(_record_episode_hash(r), r.replica_id): r for r in rl_records}
    common = sorted(set(bc_by) & set(rl_by))
    deltas: Dict[Tuple[str, int], int] = {}
    for key in common:
        b, rl = bc_by[key], rl_by[key]
        if (b.rng_contract_version != rl.rng_contract_version
                or b.policy_noise_seed != rl.policy_noise_seed):
            raise ValueError(
                f"RNG provenance mismatch on paired episode {key}: BC "
                f"(contract {b.rng_contract_version!r}, seed "
                f"{b.policy_noise_seed}) vs RL (contract "
                f"{rl.rng_contract_version!r}, seed {rl.policy_noise_seed})"
                " — the two sides must have drawn the identical noise "
                "stream"
            )
        if b.episode_config != rl.episode_config:
            raise ValueError(
                f"episode config mismatch on paired episode {key}: BC "
                f"{b.episode_config} vs RL {rl.episode_config} — same "
                "episode hash must mean the same frozen episode"
            )
        deltas[key] = (
            int(bool(rl.rl["success"])) - int(bool(b.bc["success"]))
        )
    unpaired_bc = len(set(bc_by) - set(rl_by))
    unpaired_rl = len(set(rl_by) - set(bc_by))
    if unpaired_bc or unpaired_rl:
        raise ValueError(
            f"unpaired episodes: {unpaired_bc} BC-only, {unpaired_rl} "
            "RL-only — the paired matrix must cover the same episode "
            "keys on both sides"
        )
    success_deltas = tuple(deltas[key] for key in common)
    return_deltas = tuple(
        float(rl_by[key].rl["return"]) - float(bc_by[key].bc["return"])
        for key in common
    )
    mean_delta = (
        float(sum(success_deltas) / len(success_deltas))
        if success_deltas else 0.0
    )
    return PairedAggregate(
        common_n=len(common),
        unpaired_bc=unpaired_bc,
        unpaired_rl=unpaired_rl,
        per_key=deltas,
        success_deltas=success_deltas,
        return_deltas=return_deltas,
        mean_success_delta=mean_delta,
        all_zero=all(d == 0 for d in success_deltas),
    )


def validate_paired_matrix(
    records: Any,
    expected_entries: int = 12,
    expected_replicas: int = 5,
) -> None:
    """Matrix completeness gate (§11.5.5).

    A 12x5 matrix missing one replica is INCOMPLETE — the artifact
    cannot claim the G8 sample; duplicate (episode_config_hash,
    replica_id) keys hard-fail first (a duplicated key is not an
    incomplete matrix, it is a corrupt one).
    """
    seen = set()
    entries = set()
    per_entry: Dict[str, set] = {}
    for record in records:
        ep_hash = _record_episode_hash(record)
        key = (ep_hash, record.replica_id)
        if key in seen:
            raise ValueError(
                f"duplicate paired episode key {key} — each "
                "(episode_config_hash, replica_id) may appear once in "
                "the paired matrix"
            )
        seen.add(key)
        entries.add(ep_hash)
        per_entry.setdefault(ep_hash, set()).add(record.replica_id)
    if (len(entries) != expected_entries
            or any(len(reps) != expected_replicas
                   for reps in per_entry.values())):
        raise ValueError(
            "paired matrix incomplete: expected "
            f"{expected_entries} entries x {expected_replicas} replicas, "
            f"got {len(entries)} entries with per-entry replica counts "
            f"{sorted(len(reps) for reps in per_entry.values())}"
        )


def paired_eval_report(
    bc_records: Any, rl_by_seed: Dict[str, Any], *, bank_id: Optional[str] = None,
    bc_checkpoint_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """The report aggregator boundary (§11.5.5).

    bc_records = the BC baseline artifacts' episodes (deduplicated
    across RL artifacts — the counterexample input); rl_by_seed maps a
    seed label to its policy's episodes. Every section carries the full
    three-n aggregation (raw n / unique n / cluster n — never a single
    n); the BC section pools the BC-side outcomes, RL sections the
    RL-side outcomes; "paired" carries the per-seed paired deltas vs
    the BC reference.

    R6-EVAL-1 (§11.5.1): bank_id is threaded into every aggregation so
    the R5.4 RNG-provenance check RUNS — without it the derived
    policy_noise_seed re-derivation is silently skipped and a tampered
    record would aggregate as if clean. Records without a mirrored
    policy_checkpoint_hash fall back to the aggregation-level hash only
    when every record shares one; the hard-fail in
    deduplicate_policy_records applies when neither exists.
    bc_checkpoint_hash normalizes the BC section's unit identity to the
    shared BC reference (the 18/24 counterexample fix at the report
    boundary — the RL artifacts' BC episodes carry RL hashes).
    """
    # None = caller has no bank identity (no RNG re-derivation);
    # a non-None value must be non-empty — the R5.4 check re-derives
    # every policy_noise_seed from it, so an empty id is a silent
    # provenance break (same contract as deduplicate_policy_records).
    if bank_id is not None and (not isinstance(bank_id, str) or not bank_id):
        raise ValueError(
            "paired_eval_report requires a non-empty bank_id — the R5.4 "
            "RNG-provenance check re-derives every policy_noise_seed "
            "from it"
        )
    report: Dict[str, Any] = {
        "bc": aggregate_binary_outcomes(bc_records, side="bc",
                                        bank_id=bank_id,
                                        bc_checkpoint_hash=bc_checkpoint_hash),
    }
    paired: Dict[str, Any] = {}
    for label, records in rl_by_seed.items():
        report[label] = aggregate_binary_outcomes(records, side="rl",
                                                  bank_id=bank_id)
        pair = paired_policy_delta(bc_records, records)
        paired[label] = {
            "common_n": pair.common_n,
            "unpaired_bc": pair.unpaired_bc,
            "unpaired_rl": pair.unpaired_rl,
            "mean_success_delta": pair.mean_success_delta,
            "all_zero": pair.all_zero,
        }
    report["paired"] = paired
    return report
