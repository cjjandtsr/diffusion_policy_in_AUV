# Session workspace instructions

## This machine's WSL loopback (`lo`) is unreliable — never use `127.0.0.1` for ROS

Verified 2026-11-13. WSL routes TCP to `127.0.0.1` out to the Windows-side loopback relay
(`127.0.0.1 via 169.254.73.152 dev loopback0 table 127`), so the whole `lo` interface
drops or refuses connections intermittently (self-connect 2/10, primary NIC 10/10).
`ping 127.0.0.1` still succeeds, so ICMP proves nothing. `roslaunch` has only a 5 s
self-check window (`roslaunch/server.py:404`), therefore
`RLException: Unable to contact my own server at [http://127.0.0.1:<port>]` and
`Failed to contact master at [127.0.0.1:...]` appear intermittently and retries never help.

Always start ROS with the primary interface address, re-detected on every run:

```bash
ROS_HOST_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
[ -n "$ROS_HOST_IP" ] || ROS_HOST_IP=$(hostname -I | awk '{print $1}')
export ROS_MASTER_URI=http://$ROS_HOST_IP:$PORT
export ROS_IP=$ROS_HOST_IP
```

Never use `localhost`, `127.0.0.1`, `ROS_HOSTNAME=localhost`, or `hostname -I | head -1`
(the first address may be `10.255.255.254`, the WSL DNS proxy bound to `lo`).
Never wrap ROS commands in `stdbuf -oL` (it injects `libstdbuf.so` into `LD_PRELOAD` and
breaks the master self-check); use `PYTHONUNBUFFERED=1`.

Full note (symptoms, self-check snippet, affected scripts):
`/home/cjj/.claude/rules/wsl-ros-network.md`.
