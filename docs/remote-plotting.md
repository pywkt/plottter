# Remote Plotting (Wireless)

Plot to a plotter that's connected to a small networked device — typically a Raspberry
Pi — instead of cabling the plotter to the machine you design on. Plottter sends the job
over your network; the device drives the plotter. Your computer is then **free during long
plots**, and the plotter can live wherever is convenient (another room, a shelf) without a
computer tethered to it.

This is the device side of a companion project, **`plottter-daemon`** — a small
stdlib-only HTTP service that receives jobs and drives the plotter with `pyaxidraw`. It
lives in its own repository; see its `README.md` for full Raspberry Pi setup. This guide
covers using the feature from Plottter.

> Works with any AxiDraw-compatible plotter (AxiDraw, **Uunatek iDraw H SE**, etc.) — the
> same hardware the [Direct AxiDraw Control](export-and-plotting.md#direct-axidraw-control-usb)
> path supports.

---

## How it works

```
Your computer (Plottter)                Raspberry Pi (plottter-daemon)
  Plot with AxiDraw  ──HTTP (LAN/Wi-Fi/VPN)──▶  receives SVG + settings
  (device set in Preferences)                   drives the plotter via pyaxidraw
  polls status, shows progress  ◀───────────    plots autonomously
```

- **The daemon owns the job.** Once a plot starts it runs inside the daemon, so you can
  disconnect, put your computer to sleep, or leave the network — the plot keeps going.
- **One job at a time**, with pause / resume / stop and live progress, exactly like the USB
  dialog.
- **Your design is never altered.** The daemon plots the SVG verbatim with path reordering
  disabled — all optimization happens in Plottter.
- **Feature parity with USB.** Pen up/down calibration, plotting, pause/resume, and
  multi-layer pen-swap all work over the network.

---

## 1. Set up the daemon (on the Pi)

Full instructions are in the `plottter-daemon` project's `README.md`. In short, on a
Raspberry Pi (or any always-on Linux host) connected to the plotter by USB:

```bash
# in the plottter-daemon folder, on the Pi
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
pip install https://cdn.evilmadscientist.com/dl/ad/public/AxiDraw_API.zip
python -m plottter_daemon --device-name "iDraw H SE A2"   # prints a token to copy
```

- It listens on port **8080** by default and prints an access **token** (paste it into
  Plottter). Use `--no-auth` instead if you don't want a token on a trusted home network.
- Make sure the Pi user can reach the serial port (`groups` should include `dialout`).
- Run it under **systemd** (`deploy/plottter-daemon.service`) so it starts on boot and you
  never need an SSH session open.

Confirm it's reachable without hardware first by running with `--fake --no-auth` and
connecting from Plottter (below) — handy for testing the network path.

---

## 2. Configure Plottter

Open **Edit → Preferences** and find the **Remote Plotter** group:

1. **Device URL** — `http://<pi-host-or-ip>:8080` (e.g. `http://plotter.local:8080`, a LAN
   IP, or a Tailscale name — anything reachable).
2. **Token** — paste the token the daemon printed (leave blank if you ran it `--no-auth`).
3. Tick **Enable remote plotting**, then click **Test Connection** to confirm the daemon is
   reachable. Click **OK** to save.

These settings are remembered between sessions. Untick **Enable remote plotting** any time to
plot locally over USB instead.

Now open **Plot with AxiDraw**. The connection indicator at the top tells you what's
connected:

- *● Connected via network — \<device\>* → plotting goes to the Pi.
- *● Connected via USB — \<device\>* → remote plotting is off (or unreachable), so it falls
  back to a USB-connected plotter.

Hit **Refresh** in the dialog to re-check the connection after starting the daemon or changing
the Preferences.

> **Set the model.** The job carries your **Model** selection (e.g. *AxiDraw SE/A2* for the
> iDraw H SE A2), so set it the same as you would for USB — that's what tells the daemon the
> plot-bed size.

---

## 3. Plot

Everything in the dialog now talks to the remote plotter:

- **Calibrate** with **Raise Pen / Lower Pen** — these go over the network to the real
  plotter, so set your pen-up/down positions just as you would over USB.
- **Plot Now** sends the job. Progress updates as it plots; you can then **disconnect or step
  away** — the Pi finishes the plot on its own.
- **Pause / Resume** and **Stop** work mid-plot. **Stop** also clears the job on the device, so
  use it (rather than just closing) when you don't intend to resume.
- **Multi-color (pen swap):** with "Pause between layers" enabled, each layer is sent as its
  own job and you're prompted to swap the pen between them — you'll need to be present for the
  swaps, as with USB.

### Reconnecting after a dropped connection

Because the device owns the job, a brief Wi-Fi blip is invisible — Plottter keeps following the
plot once the connection returns. If the connection drops for longer, the plot keeps running on
the device and the dialog treats it as paused so you can pick it back up.

To reconnect, just **reopen Plot with AxiDraw** (or hit **Refresh**). If the device still has a
job, the dialog detects it and offers to **Resume** (if it paused) or **Stop** it — so a plot
left paused after a disconnect never strands the plotter as "busy." You should no longer need to
restart the daemon to get unstuck. If you ever do see a *busy* message starting a new plot,
Plottter will offer to stop the leftover job for you.

---

## Security

Auth is optional and independent of how your network is set up (LAN, VPN, Tailscale — the URL
field accepts any of them):

- The daemon can require a **bearer token** on all control endpoints; `/health` stays open so
  Plottter can detect it. Paste the token into the **Token** field.
- On a trusted home network you can run the daemon with `--no-auth` and leave the token blank.
- The daemon physically moves hardware, so don't expose it directly to the public internet —
  keep it on your LAN/VPN.

---

## Troubleshooting

**Status shows "No plotter daemon reachable at …".**
- Is the daemon running on the Pi? (`systemctl status plottter-daemon`, or check your terminal.)
- Is the URL right, including `http://` and `:8080`? Try the Pi's IP instead of its hostname.
- Can your computer reach the Pi at all? `ping <pi-ip>`. Same network / VPN?

**Connected, but plotting errors with "Plotter not found".**
- On the Pi, confirm the plotter enumerated: `ls /dev/ttyUSB*` (and `dmesg | tail` after
  plugging in). Make sure the daemon's user is in the `dialout` group.
- If a port exists but auto-detect fails, start the daemon with `--serial-port /dev/ttyUSB0`.

**"The remote plotter is busy."**
- One job at a time. Wait for the current plot to finish (or Stop it) before sending another.

**The plot kept going after I disconnected.**
- That's by design — the daemon owns the job. Use **Stop** (or the plotter's physical pause)
  to halt it.

See also [Direct AxiDraw Control](export-and-plotting.md#direct-axidraw-control-usb) for the
USB path and the shared pen/speed settings, and the `plottter-daemon` README for Pi-side
setup and systemd.
