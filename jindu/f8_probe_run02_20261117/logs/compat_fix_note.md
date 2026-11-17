# F8 v1.2 compat fix (recorded, not silent)
date: 2026-11-16T22:08:54-06:00
reason: load_override(current_root) executes e5_runtime.py, which file-loads
  <current_root>/third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py
  when min_std_vector is supplied. That companion file is not shipped in F8_probe_code_v1_2.zip.
action: copied the F7-current vendor companion from BASE (no edits).
source: /home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py
dest:   /home/cjj/AUV/program/f8_probe_code_20261116_v12/third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py
8f495b48d98180f1932161f188c014c40f3b408975c512794bb1d5975575aac4  source diffusion_vpg.py
8f495b48d98180f1932161f188c014c40f3b408975c512794bb1d5975575aac4  copied diffusion_vpg.py
proof of neutrality:
 - package uav_idp3/e5_runtime.py == BASE e5_runtime.py == bade7d2a67fe8b4a604ff1387a294d42a176af2a458c2da450e497c61b9f4254
 - BASE vs NATIVE vendor: 103 .py files, only 2 differ (diffusion_vpg.py, e5_runtime.py); other 101 identical
 - copied diffusion_vpg.py imports only model.diffusion.diffusion and model.diffusion.sampling (identical in both trees)
 - no package file was modified; the zip stays as delivered
