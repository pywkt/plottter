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
  "Remote Plotter (network)"                    drives the plotter via pyaxidraw
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

Open **Plot with AxiDraw** and find the **Remote Plotter (network)** group:

1. **Device URL** — `http://<pi-host-or-ip>:8080` (e.g. `http://plotter.local:8080`, a LAN
   IP, or a Tailscale name — anything reachable).
2. **Token** — paste the token the daemon printed (leave blank if you ran it `--no-auth`).
3. Tick **Send to remote device**, then click **Refresh connection**.

The status line at the top of the dialog tells you what's connected:

- *Connected: \<device\> (network) — \<url\>* → plotting goes to the Pi.
- *pyaxidraw is installed…* / *USB* → the network device wasn't selected/reachable, so it
  falls back to a USB-connected plotter.

These settings are remembered between sessions. Untick **Send to remote device** any time to
plot locally over USB instead.

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
- **Pause / Resume** and **Stop** work mid-plot.
- **Multi-color (pen swap):** with "Pause between layers" enabled, each layer is sent as its
  own job and you're prompted to swap the pen between them — you'll need to be present for the
  swaps, as with USB.

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
