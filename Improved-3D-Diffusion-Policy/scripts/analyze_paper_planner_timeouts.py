"""Read adjudicated episodes; describe distance/timeout association without reruns."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import numpy as np


def summarize(rows):
    tasks={r['task_id']:r['initial_goal_distance_m'] for r in rows}
    cuts=np.quantile(list(tasks.values()),[.25,.5,.75])
    groups=defaultdict(list)
    for r in rows:
        groups[r['policy'],int(np.searchsorted(cuts,r['initial_goal_distance_m'],side='right'))].append(r)
    result=[]
    for (policy,group),rs in sorted(groups.items()):
        timeouts=[r for r in rs if r['paper_done_reason'] in ('time_limit','timeout')]
        progress=[(r['initial_goal_distance_m']-r['min_goal_distance_m'])/r['initial_goal_distance_m'] for r in timeouts]
        failures=[r['planner_failed_plans_cumulative'] for r in timeouts if r.get('planner_failed_plans_cumulative') is not None]
        result.append(dict(policy=policy,distance_quartile=group+1,tasks=len({r['task_id'] for r in rs}),
            episodes=len(rs),outcomes=dict(Counter(r['paper_done_reason'] for r in rs)),
            timeout_episodes=len(timeouts),timeout_rate=len(timeouts)/len(rs),
            timeout_median_normalized_best_progress=float(np.median(progress)) if progress else None,
            timeout_failure_counter_available=len(failures),
            timeout_median_failed_plans=float(np.median(failures)) if failures else None))
    return dict(task_distance_quartiles_m=cuts.tolist(),groups=result,
                limitation='Descriptive association only: long distance does not establish that a different deadline would avoid failure. Original deadline/results unchanged.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[json.loads(line) for path in a.records for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(summarize(rows),f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
