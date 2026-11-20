# convert_v5_to_v65.py
import zarr, shutil, numpy as np, os

SRC = '/home/cjj/AUV/avoidance_dataset/merged_body_11.zarr'
DST = '/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr'

if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

dst = zarr.open(DST, mode='a')
old_state = dst['data/state'][:]  # (N, 15)

# v5 下标: pos[0:3] + goal_body[9:12] + dist[12] + sin_yaw[13] + cos_yaw[14]
V65_IDX = [0, 1, 2, 9, 10, 11, 12, 13, 14]
new_state = old_state[:, V65_IDX].astype(np.float32)

del dst['data/state']
dst['data'].create_dataset('state', data=new_state, chunks=(1000, 9), dtype='float32')
dst.attrs['state_version'] = 'v6.5'
dst.attrs['state_layout'] = 'pos_world(3) + goal_body(3) + dist(1) + sin_yaw + cos_yaw'

print(f'[DONE] state shape: {new_state.shape}')

from sklearn.linear_model import LinearRegression
action = dst['data/action'][:]
r2 = LinearRegression().fit(new_state, action[:,1]).score(new_state, action[:,1])
print(f'[VERIFY] R² → Δpy = {r2:.4f}  (预期 0.7018)')