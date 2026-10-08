# Keeping the engine online

The website (Vercel) is always on. The engine, which runs the models, needs a machine with about 1.5 GB
of free memory and the 440 MB of model files, which is why it cannot live on Vercel. These are the
options, cheapest first. Facts about providers were checked on 8 October 2026; they change, so check
again before you rely on one.

| Option | Cost | Always on? | Notes |
|---|---|---|---|
| Laptop + tunnel (`scripts/go_live_tunnel.ps1`) | free | only while the laptop runs | What the project uses for demos. On the campus network the link changes about every 20 minutes (localhost.run); on a phone hotspot Cloudflare works and the link lasts for hours. |
| Oracle Cloud Always Free, Ampere A1 VM | free | yes, with conditions | Since 15 June 2026 the free Arm allowance is 2 cores and 12 GB of memory (it was 4 and 24). Enough for the engine. Oracle may reclaim a free VM that stays idle for 7 days (CPU, network and memory all under 20%). Sign-up asks for a payment card to verify identity. |
| Any small paid VM (2 vCPU, 4 GB) | low monthly cost | yes | The most predictable option for a pilot. |
| Hugging Face Docker Space (`deploy/huggingface`) | needs a paid plan | yes | The project's Dockerfile is ready, but Docker Spaces asked for PRO when we tried (error 402). |

## Oracle Always Free, step by step

1. Create the account, then **Compute → Instances → Create instance**. Shape: *Ampere*
   `VM.Standard.A1.Flex`, 2 OCPU, 12 GB. Image: Ubuntu 24.04. Download the SSH key it offers.
2. In the instance's subnet, add an ingress rule for TCP 8001, or put Caddy in front for HTTPS (below).
3. On the VM:

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone -b audit-2026-10-03 https://github.com/udbhav968-creator/SIH_PROJECT.git && cd SIH_PROJECT
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m scripts.fetch_cnn_backbone          # the large weights that are not in git
export ROAD_SHIELD_PUBLIC=1 ROAD_SHIELD_FLEET_KEY=<secret> ROAD_SHIELD_SEAL_KEY=<another secret>
python -m api.server 8001
```

4. HTTPS (browsers block a plain-HTTP engine from the Vercel site): install Caddy and use a free
   hostname such as `<name>.duckdns.org` pointed at the VM, with this `/etc/caddy/Caddyfile`:

```
<name>.duckdns.org {
    reverse_proxy 127.0.0.1:8001
}
```

5. Make it start on boot with a systemd service (`ExecStart=/home/ubuntu/SIH_PROJECT/.venv/bin/python -m api.server 8001`,
   the same `Environment=` lines, `Restart=always`).
6. Add the new address to the site's allow-list in `web/app.js` (`engineUrl`, the `allowed` pattern)
   and to `web/config.js`, then push and merge.

## Keys

| Variable | What it protects | Where it must match |
|---|---|---|
| `ROAD_SHIELD_API_KEY` | write endpoints (set automatically in public mode) | callers that write |
| `ROAD_SHIELD_FLEET_KEY` | bus packets (AES-256-GCM) | server and every bus |
| `ROAD_SHIELD_SEAL_KEY` | work-order seals (HMAC-SHA256) | server and whoever verifies orders |

Keys never go in git, chat or screenshots. A new random key: `python -c "import secrets; print(secrets.token_hex(32))"`.
