# Remote Optimization

The full **Optimize** pipeline (nearest-neighbour reorder + 2-opt + 3-opt +
Or-opt, plus weld / simplify / merge / join / clip / filter preprocessing) is
the heaviest CPU work Plottter does. On big plots — dense maps, fine pixel
art, stippled portraits — a thorough optimize pass can take many seconds to
several minutes on a modest laptop.

Plottter's **remote optimization** feature lets you keep the GUI on your usual
laptop while shipping just the heavy compute to a faster machine on your
network. No display forwarding, no `ssh -X` lag — only the polylines travel
over the wire, then the optimized polylines come back.

---

## How it works

The GUI serialises the active layer's paths and the dialog settings as JSON,
spawns `ssh <host> plottter --optimize`, and pipes the payload to the remote
process's stdin. The remote runs the same `run_optimization_pipeline()`
function the local GUI uses, streams progress as line-buffered JSON on stderr,
and writes the optimized paths as JSON to stdout. The local worker parses
that and applies the result via the normal `set_layer_paths` undo-aware path.

Because the same pipeline function is used in all three places (local
QThread, CLI, remote), local and remote runs produce identical output for
identical inputs.

---

## Setup

### 1. Install Plottter on the fast machine

A normal install. SSH into it and run:

```bash
git clone https://github.com/pywkt/plottter.git
cd plottter
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You can also install the `[fast]` extra for the numba JIT (~2–3× faster
weld/merge), which compounds with the remote-CPU win:

```bash
pip install -e ".[fast]"
```

### 2. Make `plottter` reachable over SSH

SSH command sessions (`ssh host plottter ...`) use a stripped-down `PATH`
that does **not** include venv binaries. You have two options:

**Option A — Symlink to a system PATH directory** (one-time, requires sudo):

```bash
sudo ln -sf "$(pwd)/.venv/bin/plottter" /usr/local/bin/plottter
```

Note the **absolute path** in the link target. `./venv/bin/plottter` would
create a dangling symlink that only works when you happen to be in the repo
directory.

Verify from your laptop:

```bash
ssh <host> plottter --help | head -3
```

Should print the help banner.

**Option B — Use the "Remote Command" override** (no sudo, more robust):

Skip the symlink entirely. In Plottter's Preferences → Remote Optimization,
put the absolute path to the venv binary into the **Remote Command** field
(see *Preferences fields* below). The worker will run that absolute path
directly instead of relying on `PATH`.

### 3. Configure Plottter

**Edit → Preferences → Remote Optimization:**

| Field | What to put | Example |
|---|---|---|
| Remote Host | Anything `ssh` understands: a hostname, `user@host`, or an alias from `~/.ssh/config` | `catherine@fastbox` or `fastbox` |
| Remote Command | Leave blank to use `plottter` (Option A above). Otherwise the absolute path to the venv binary | `/home/catherine/plottter/.venv/bin/plottter` |

Leave **Remote Host** blank to be prompted on every Optimize Remotely click
(handy if you switch between machines).

### 4. Run it

- **Tools → Optimize Current Layer Remotely…**
- The settings dialog is the same one you'd get for a local optimize. Pick
  your stages, click OK.
- A progress dialog appears; cancellation kills the SSH process, which
  propagates SIGTERM to the remote `plottter`.
- On finish, the status bar reports the travel/lift delta:
  ```
  Remote optimize: 'Roads' travel 18342.1 → 9217.3 mm  ·  lifts 412 → 117
  ```

---

## Lower latency for repeat calls — SSH ControlMaster

Each fresh `ssh` invocation costs a TCP handshake plus key exchange — a
couple hundred milliseconds even on a fast LAN. For long optimizes that's
noise; for batched short ones it adds up.

`ControlMaster` lets all subsequent SSH calls reuse a single persistent
connection:

```ssh-config
# ~/.ssh/config (on your laptop)
Host fastbox
    HostName 100.89.227.87
    User catherine
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 5m
```

Set **Remote Host** in Preferences to just `fastbox`. The first call sets up
the multiplexed master (~250 ms once); subsequent calls reuse it in ~5 ms.
After 5 minutes of inactivity the master closes itself.

---

## Tailscale SSH — zero open ports

If your fast box is on [Tailscale](https://tailscale.com), you can use
**Tailscale SSH** to skip running OpenSSH (`sshd`) on the remote entirely.
The Tailscale daemon (`tailscaled`) handles authentication via your Tailscale
identity and ACLs — no port 22 forward, no public SSH server, no firewall
hole.

This means you can run remote optimize **from anywhere** (coffee shop,
office, phone tether) without exposing any port on the fast box to the
public internet. Tailscale's encrypted mesh routes the connection over
WireGuard.

**Setup:**

1. Install Tailscale on both machines and log them into the same tailnet:
   <https://tailscale.com/download>
2. On the fast box, enable Tailscale SSH:
   ```bash
   sudo tailscale up --ssh
   ```
   Then disable / firewall off the regular SSH server if you want
   (`sudo systemctl disable --now sshd` on systemd distros; on Debian/Ubuntu
   the unit may be `ssh.service`).
3. In Plottter's Preferences, set **Remote Host** to the fast box's
   Tailscale IP (the `100.x.y.z` address shown by `tailscale ip`) or its
   MagicDNS hostname (`fastbox.your-tailnet.ts.net`).

Plottter's worker just calls `ssh`, so it transparently uses whichever route
Tailscale set up — no Plottter changes needed. You get the same speedup
with strictly less exposed surface area than a typical SSH setup.

**Caveat:** Tailscale ACL changes can break in-flight connections. If a
remote optimize stops mid-run after an ACL edit, just re-run it.

---

## CLI subcommand reference

The remote-optimize feature is built on a CLI mode you can also use
directly for scripting / batch / shell pipelines.

```bash
plottter --optimize < job.json > result.json
```

### Input (stdin, single JSON object)

```json
{
  "paths": [[[x, y], [x, y], ...], ...],
  "settings": {
    "run_weld": false, "weld_tolerance": 0.1,
    "run_simplify": true, "simplify_tolerance": 0.1,
    "run_filter": true, "filter_min_length": 0.5,
    "run_clip": true,
    "run_merge": true, "merge_threshold": 0.5,
    "run_join": false, "join_threshold": 0.1,
    "run_2opt": true, "run_3opt": false, "run_or_opt": true,
    "num_starts": 5
  },
  "clip_bounds": [x1, y1, x2, y2],
  "generator_info": null
}
```

`settings` keys you omit fall back to defaults. `clip_bounds` is required if
`run_clip` is true. `generator_info` is optional; if you pass
`{"_generator_name": "Map"}` the pipeline force-enables the Join stage to
match the per-layer policy for map layers.

### Output (stdout, single JSON object)

```json
{
  "paths": [[[x, y], [x, y], ...], ...],
  "before_travel": 18342.1,
  "after_travel": 9217.3,
  "before_lifts": 412,
  "after_lifts": 117
}
```

### Progress (stderr, one JSON per line)

```
{"progress": 35}
```

Safely ignored — but useful if you're driving a progress bar.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success — result JSON is on stdout |
| 1 | Invalid JSON on stdin, missing fields, or an exception in the pipeline |

---

## Troubleshooting

### "command not found: plottter"

Non-interactive SSH sessions use a minimal `PATH`. Either symlink the venv
binary to `/usr/local/bin` (with an **absolute** target path — see *Setup
step 2*) or use the **Remote Command** override field with the absolute
venv path.

### Qt platform plugin "xcb" error / "could not connect to display"

The remote `plottter` was launched without `--optimize` being recognised as
a CLI flag, so it fell through to the GUI startup path and crashed because
no display is available. This means the version on the fast box is older
than the one that introduced `--optimize`. Pull the latest code on the
fast box; editable installs (`pip install -e .`) pick up pure-Python file
changes without re-running install.

### Hangs forever / silent timeout

Likely an interactive password prompt that your SSH key auth isn't
covering. Run `ssh <host> plottter --help` from a terminal to confirm
non-interactive auth works. If it asks for a password, set up SSH keys or
load your agent (`ssh-add ~/.ssh/id_ed25519`).

### Speeds are no better than local

A few possibilities, in order of likelihood:

1. The remote machine isn't actually faster, or it's busy. Run `htop` /
   `btop` on the remote while optimizing — you should see one core at
   100 %.
2. The optimize is dominated by stages that don't scale linearly with CPU
   speed (e.g. a huge `merge` step is bound by the SciPy KD-tree, not by
   raw Python). Try toggling stages off in the dialog.
3. The payload is tiny and the SSH handshake dominates. Enable
   `ControlMaster` (above), or just accept that remote is for big jobs.

### "Remote optimize failed (exit 127)"

The remote shell couldn't find the `plottter` command at all — see the
"command not found" troubleshooting above.

---

## Security notes

- SSH key authentication is recommended over password auth so the worker
  doesn't block on a prompt. Use `ssh-add` to load your key into the agent
  before launching Plottter, or set up agent forwarding if you bounce
  through a jump host.
- The wire payload is just polylines and optimize settings — no project
  metadata, no API keys, no canvas state. Still, run remote-optimize only
  against machines you trust.
- If you use Tailscale SSH, the remote optimize inherits Tailscale's ACL
  model — only nodes/users your ACL allows can reach the `plottter` binary.
- The remote process inherits the remote user's shell environment. Don't
  point Plottter's Remote Command at a binary you don't fully trust.
