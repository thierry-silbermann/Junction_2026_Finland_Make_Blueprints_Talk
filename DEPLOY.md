# Deploying PlanSight on a Hetzner VPS

This guide puts the PlanSight app (`drawingqa serve`: the Live blueprint and Benchmark tabs) and the static review report online, on one Hetzner Cloud server with HTTPS and a password.

The server does not need a GPU. Inference goes to NVIDIA's hosted Nemotron API, or to your own vLLM server if you rent one (see [Optional: self-hosted Nemotron](#optional-self-hosted-nemotron)). The VPS renders drawings, reads DXF and IFC files, and serves the web app.

```
browser ──HTTPS──► Caddy :443 (TLS + password) ──► drawingqa serve 127.0.0.1:8000 ──► NVIDIA API
                        └─► /report/  static review report
```

Only need it online for a couple of hours, with no domain? Do steps 1–6, then follow [Quick option: a 2-hour demo over plain HTTP](#quick-option-a-2-hour-demo-over-plain-http) instead of steps 7 and 8.

## 1. Create the server

In the Hetzner Cloud Console, choose **Add Server**:

| Setting | Choice |
|---|---|
| Location | Helsinki or Falkenstein / Nuremberg (EU, close to the jury) |
| Image | Ubuntu 24.04 |
| Type | Shared vCPU, **x86** (CX or CPX line), at least **4 vCPU / 8 GB RAM / 80 GB disk** (for example CX32 or CPX31) |
| Networking | Public IPv4 + IPv6 |
| SSH key | Add your public key (`cat ~/.ssh/id_ed25519.pub`) |
| Firewall | Create one: allow inbound TCP 22, 80, 443 and ICMP; nothing else |

Pick **x86, not Arm (CAX)**: `ifcopenshell` and `PyMuPDF` ship ready-made wheels for x86 Linux. The dataset is about 11 GB, so a 40 GB disk is too tight once rendered pages and uploads accumulate.

**Domain.** Point an `A` record (and an `AAAA` record for IPv6) at the server's IP, for example `plansight.example.com`. If you have no domain, `<IP-with-dashes>.sslip.io` works as a real hostname for Let's Encrypt. For example, `65-21-10-4.sslip.io` resolves to `65.21.10.4`.

## 2. Prepare the server

```bash
ssh root@SERVER_IP

apt update && apt -y upgrade
apt -y install python3 python3-venv python3-dev git rsync ufw debian-keyring debian-archive-keyring apt-transport-https curl

# Host firewall (in addition to the Hetzner Cloud Firewall)
ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable

# A dedicated user that owns the app
adduser --system --group --home /opt/plansight plansight
```

Ubuntu 24.04 ships Python 3.12; PlanSight needs 3.11 or newer.

## 3. Get the code

The repository is public on GitHub:

```bash
sudo -u plansight git clone https://github.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk.git /opt/plansight/app
```

If the repository is not pushed yet, copy the working tree from your laptop instead, from the repo root:

```bash
rsync -az --exclude '.venv*' --exclude data --exclude runs --exclude '.git' --exclude '.env' \
  ./ root@SERVER_IP:/opt/plansight/app/
ssh root@SERVER_IP chown -R plansight:plansight /opt/plansight/app
```

## 4. Install

```bash
cd /opt/plansight/app
sudo -u plansight python3 -m venv .venv
sudo -u plansight .venv/bin/pip install --upgrade pip
sudo -u plansight .venv/bin/pip install -r requirements.txt -r requirements-dxf.txt -r requirements-ifc.txt -r requirements-app.txt
sudo -u plansight .venv/bin/drawingqa --help     # sanity check
```

The DXF and IFC extras are what make the zero-token routes work: without them, Swiss and IFC drawings fall back to Nemotron. You don't need the detector environment (`.venv-detector`, PyTorch, Ultralytics) on the server.

## 5. Copy the data, the benchmark run and the cache

None of these are in git. Copy them from your laptop, from the repo root:

```bash
# Drawings + questions (~11 GB; resumable, rerun if interrupted)
rsync -az --info=progress2 data/ root@SERVER_IP:/opt/plansight/app/data/

# The benchmark run shown in the Benchmark tab (~19 MB)
rsync -az runs/per-folder-6-v2 root@SERVER_IP:/opt/plansight/app/runs/

# Optional: the warm caches, so every sample answers instantly without new API calls
rsync -az .drawingqa/ root@SERVER_IP:/opt/plansight/app/.drawingqa/
rsync -az runs/live/   root@SERVER_IP:/opt/plansight/app/runs/live/

ssh root@SERVER_IP chown -R plansight:plansight /opt/plansight/app
```

To save disk space and upload time, you can copy only the folders the demo uses instead of all of `data/`: `questions.jsonl`, `cubicasa5k`, `swiss_dwellings`, `ifc_models`, `opening_design`, `ifc_bench`, `finna` and `europeana`. Sample tiles for missing folders simply won't appear.

Before the jury sees it, re-verify the 10 examples in the Live tab on the server itself (after step 6). The live ones call Nemotron for real, so this confirms that the key and the network work:

```bash
cd /opt/plansight/app && sudo -u plansight .venv/bin/python scripts/curate_showcase.py
```

If you skip the cache copy, you can warm it on the server after step 6 (this calls the NVIDIA API once per sample):

```bash
cd /opt/plansight/app && sudo -u plansight .venv/bin/python scripts/warm_demo.py --model nemotron_nvidia
```

## 6. Configure the API key

```bash
cd /opt/plansight/app
sudo -u plansight install -m 600 /dev/null .env
sudo -u plansight nano .env
```

```dotenv
NVIDIA_API_KEY=nvapi-...
```

The app reads `.env` from its working directory, which is why the service below sets `WorkingDirectory`. Keep it `chmod 600`; never commit it.

## Quick option: a 2-hour demo over plain HTTP

This path needs no domain and no certificate. The app stops by itself after 2 hours, and you then delete the server. Do steps 1–6 first, then pick one of the two variants below.

What you give up:

- **Encryption.** Traffic, including uploaded drawings and the password, is unencrypted. That's fine for the public dataset drawings in the demo, but don't use it for client drawings.
- **Protection of your API key, if there's no password.** Anyone who has the URL can run detections on your NVIDIA key. Share the link only with the jury, or use variant B.

### Variant A: the app directly on port 8000 (simplest, no password)

```bash
# Open the port: in the Hetzner Cloud Firewall add inbound TCP 8000, and on the server:
ufw allow 8000/tcp

# Run for 2 hours, then stop automatically (survives closing your SSH session)
systemd-run --unit plansight-demo --uid plansight --working-directory /opt/plansight/app \
  -p RuntimeMaxSec=2h -p Environment=PYTHONUNBUFFERED=1 \
  /opt/plansight/app/.venv/bin/drawingqa serve --host 0.0.0.0 --port 8000
```

Open `http://SERVER_IP:8000/`; the Benchmark tab is at `http://SERVER_IP:8000/#bench`.

### Variant B: behind a password on port 80

The app stays on `127.0.0.1` only, and Caddy adds a login in front of it, without HTTPS.

```bash
systemd-run --unit plansight-demo --uid plansight --working-directory /opt/plansight/app \
  -p RuntimeMaxSec=2h -p Environment=PYTHONUNBUFFERED=1 \
  /opt/plansight/app/.venv/bin/drawingqa serve --host 127.0.0.1 --port 8000
```

Install Caddy and create a password hash as in [step 8](#8-https-a-password-and-the-static-report-caddy). Then use this `/etc/caddy/Caddyfile`; the `:80` site address tells Caddy to serve plain HTTP on any hostname or IP, with no certificate:

```caddy
:80 {
	basic_auth {
		jury PASTE_THE_HASH_HERE
	}

	handle_path /report/* {
		root * /opt/plansight/app/project_submission/demo
		file_server
	}

	handle {
		request_body {
			max_size 60MB
		}
		reverse_proxy 127.0.0.1:8000
	}
}
```

```bash
systemctl reload caddy
```

Open `http://SERVER_IP/` (the user is `jury`). The static report is at `http://SERVER_IP/report/`.

### During and after the demo

| Task | Command |
|---|---|
| Is it running, and how long is left? | `systemctl status plansight-demo` |
| Logs | `journalctl -u plansight-demo -f` |
| Stop early | `systemctl stop plansight-demo` |
| Run another 2 hours | Run the same `systemd-run` command again |

- **Before the jury arrives**, run `scripts/curate_showcase.py` (see step 5) to confirm the live examples answer correctly from this server.
- **Afterwards, delete the server** in the Hetzner Console (*Server → Delete*). Hetzner bills by the hour, so 2 hours costs a few cents. Take a snapshot first if you want to bring it back later. Then check your usage on [build.nvidia.com](https://build.nvidia.com/), and rotate the API key if the URL was shared beyond the jury.

## 7. Run it as a service

Create `/etc/systemd/system/plansight.service`:

```ini
[Unit]
Description=PlanSight demo app
After=network-online.target
Wants=network-online.target

[Service]
User=plansight
Group=plansight
WorkingDirectory=/opt/plansight/app
ExecStart=/opt/plansight/app/.venv/bin/drawingqa serve --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/opt/plansight/app/runs /opt/plansight/app/.drawingqa

[Install]
WantedBy=multi-user.target
```

```bash
mkdir -p /opt/plansight/app/.drawingqa && chown plansight:plansight /opt/plansight/app/.drawingqa
systemctl daemon-reload
systemctl enable --now plansight
systemctl status plansight
curl -s http://127.0.0.1:8000/api/models | head -c 300; echo
```

The app binds to `127.0.0.1` only, so the outside world reaches it exclusively through Caddy. The defaults match the local setup: config `configs/models.yaml`, data root `data`, benchmark run `per-folder-6-v2`, live output `runs/live`, and the `per_drawing` strategy. Add flags such as `--run NAME` to `ExecStart` to change them.

## 8. HTTPS, a password and the static report (Caddy)

The app has no login of its own. Anyone who reaches it can upload drawings that are sent to NVIDIA's API on your key. Put it behind a password.

```bash
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
apt update && apt -y install caddy

caddy hash-password          # type the password you will give the jury; copy the hash
```

Replace `/etc/caddy/Caddyfile` with:

```caddy
plansight.example.com {
	encode zstd gzip

	basic_auth {
		jury PASTE_THE_HASH_HERE
	}

	handle_path /report/* {
		root * /opt/plansight/app/project_submission/demo
		file_server
	}

	handle {
		request_body {
			max_size 60MB
		}
		reverse_proxy 127.0.0.1:8000
	}
}
```

```bash
systemctl reload caddy
```

Then open `https://plansight.example.com/` (the user is `jury`). Caddy obtains and renews the Let's Encrypt certificate by itself; ports 80 and 443 must be open for that. The static review report is at `/report/`, and the Benchmark tab at `/#bench`.

To make the site public instead, delete the `basic_auth` block. In that case, watch your NVIDIA API usage (see [Operating it](#operating-it)).

## 9. Check it end to end

1. The dot next to the model picker is green. This means `GET /models` on the NVIDIA API succeeded with your key.
2. **Live blueprint tab.** Click the `swiss_dwellings` sample and ask *How many doors are shown on this floor plan?* The answer should be **36**, with the badge *DXF · 0 tokens*. This proves the DXF extra and the data copy work.
3. Click a `cubicasa5k` sample and press *Detect everything*. You should see boxes within a few seconds, or instantly if the cache was copied.
4. **Benchmark tab.** You should see 805 questions, with 100% on `ifc_models` and 99.1% on `swiss_dwellings`.
5. Upload a PDF of your own; the limit is 50 MB.

## Operating it

| Task | Command |
|---|---|
| Logs | `journalctl -u plansight -f` and `journalctl -u caddy -f` |
| Restart after a config or `.env` change | `systemctl restart plansight` |
| Update the code | `cd /opt/plansight/app && sudo -u plansight git pull && sudo -u plansight .venv/bin/pip install -r requirements-app.txt && systemctl restart plansight` |
| Disk usage | `du -sh /opt/plansight/app/runs/live /opt/plansight/app/.drawingqa` |
| Clear uploads | `rm -rf /opt/plansight/app/runs/live/uploads/*` (uploads are stored by content hash; samples are unaffected) |
| API usage | [build.nvidia.com](https://build.nvidia.com/) → your account; each uncached detection is 4 requests (one per group) |

- **Backups.** Hetzner's *Backups* option (+20% of the server price) snapshots the whole disk daily. Otherwise, take a manual snapshot before the judging.
- **Resizing.** If many people use it at once, rescale the server in the Console (*Rescale*, keeping the disk size so you can go back down). The app runs PDF rendering and DXF/IFC parsing on the CPU, and it limits Nemotron calls to 2 concurrent requests (`max_concurrency` in `configs/models.yaml`).

## Optional: self-hosted Nemotron

The model picker also lists `nemotron_gpu`, which expects an OpenAI-compatible vLLM server at `http://localhost:8002/v1`. To keep drawings entirely in your own infrastructure:

1. Rent a GPU (one H100 runs the FP8 checkpoint) and start vLLM as described in `scripts/gpu/`.
2. From the VPS, open a persistent SSH tunnel, so that port 8002 on the VPS forwards to the GPU box. For example, run `autossh -N -L 8002:localhost:8000 user@GPU_HOST` as a second systemd service.
3. Choose **nemotron_gpu** in the picker; its status dot turns green once the tunnel is up.

Hetzner's own GPU servers (dedicated GEX line) can host vLLM too. Then the tunnel becomes a private-network address in `configs/models.yaml`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Red dot next to the model | Missing or wrong `NVIDIA_API_KEY` in `/opt/plansight/app/.env`, or no outbound HTTPS. Check `journalctl -u plansight`, then restart the service |
| Swiss doors come from Nemotron, not *DXF* | `ezdxf` is missing (rerun step 4) or the `.dxf` files were not copied next to the PDFs |
| Benchmark tab is empty | `runs/per-folder-6-v2` was not copied (step 5) |
| Sample tiles missing | That dataset folder is not in `data/` on the server |
| `502 Bad Gateway` | The app is not running: `systemctl status plansight` |
| Upload fails with 413 | The file is over 50 MB (the app limit) or over the Caddy `max_size` |
| Permission errors in the logs | Run `chown -R plansight:plansight /opt/plansight/app`; the service may only write to `runs/` and `.drawingqa/` |
| Certificate not issued | DNS does not point at the server yet, or port 80 is blocked in the Hetzner firewall |
