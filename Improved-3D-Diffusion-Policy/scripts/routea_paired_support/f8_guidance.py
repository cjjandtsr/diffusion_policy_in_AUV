"""F8 route points affect policy input only; the environment keeps its true goal."""
import numpy as np
from .f8_geo_field import line_clear


def route_points(world, start, goal):
    start, goal = np.asarray(start), np.asarray(goal)
    path = world.field(goal).path(start, 1e9)
    clear = lambda a, b: line_clear(world, a, b) >= .3
    one = [p for p in path if clear(start, p) and clear(p, goal)]
    if one:
        return [min(one, key=lambda p: np.linalg.norm(p-start)+np.linalg.norm(goal-p)).tolist()]
    indices = range(0, len(path), 2)
    two = [(path[i], path[j]) for i in indices for j in indices
           if j > i and clear(start, path[i]) and clear(path[i], path[j]) and clear(path[j], goal)]
    if two:
        a,b = min(two, key=lambda pair: np.linalg.norm(pair[0]-start)
                  + np.linalg.norm(pair[1]-pair[0])+np.linalg.norm(goal-pair[1]))
        return [a.tolist(), b.tolist()]
    return []


def guided_observation(observation, goal_world, spawn_offset):
    result = {k: np.asarray(v).copy() for k,v in observation.items()}
    state = result['agent_pos']
    goal = np.asarray(goal_world, np.float32)-np.asarray(spawn_offset, np.float32)
    for frame in state[-3:]:
        dx,dy,dz = goal-frame[:3]
        s,c = frame[7],frame[8]
        frame[3:7] = [c*dx+s*dy,-s*dx+c*dy,dz,np.sqrt(dx*dx+dy*dy+dz*dz)]
    return result


def entered_ball(path, point, radius=1.):
    point = np.asarray(point)
    points = np.asarray(path)
    for a,b in zip(points[:-1],points[1:]):
        delta=b-a
        t=np.clip(np.dot(point-a,delta)/max(np.dot(delta,delta),1e-12),0.,1.)
        if np.linalg.norm(a+t*delta-point) <= radius:
            return True
    return False


def advance_route(points, current, executed_path):
    while current < len(points)-1 and entered_ball(executed_path,points[current]):
        current+=1
    return current
