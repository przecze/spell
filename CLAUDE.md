# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Spellchecking ML experiment: RoBERTa fine-tuned for misspelling detection as a token classification task. Users type text into a React frontend; the API returns per-token error probabilities; the dataset explorer (Streamlit) lets you browse the training data.

## Common commands

```bash
# Local dev (API with hot-reload + Vite frontend + nginx proxy)
docker compose up backend frontend nginx
# Dev site available at http://localhost:3080

# Production gateway locally (baked frontend, WireGuard to Fly, no local model)
make gateway-up        # starts gateway + suspender-api + suspender-dataset
make gateway-down

# Deploy to production server (Ansible rsync + docker compose)
make deploy
make deploy-check      # dry-run

# Deploy Fly.io apps
make fly-deploy-api      # builds from repo root, context includes checkpoints/
make fly-deploy-dataset  # builds from dataset/ directory
make fly-deploy          # both

# Recompile API requirements (CPU-only torch, no CUDA)
make api-reqs

# WireGuard peer setup (run once per app, before first deploy)
make api-wg-create
make dataset-wg-create

# Dump all infra files to stdout / clipboard
make dump-infra
make copy-infra
```

## Architecture

### Services

| Service | Runtime | Location |
|---|---|---|
| API (FastAPI) | Python 3.13, Fly.io | `api/` |
| Dataset explorer (Streamlit) | Python 3.12, Fly.io | `dataset/` |
| Frontend (React/Vite) | TypeScript, built into gateway image | `frontend/` |
| Gateway (nginx + WireGuard) | nginx:alpine, local Docker | `gateway/`, `Dockerfile.gateway` |
| Suspenders | alpine shell scripts, local Docker | `dataset/suspender.sh` |

### Production topology

```
Internet → nginx-proxy (external) → gateway container
                                        ├── serves built frontend (static)
                                        ├── /predict → WireGuard → Fly api app
                                        └── /dataset/ → WireGuard → Fly dataset app

suspender-api   ─┐
suspender-dataset┘─ read /activity/most_recent_call.txt → Fly Machines API suspend
```

The gateway nginx uses `set $flyapp <hostname>` (variable resolver trick) so nginx starts even when the Fly machine is suspended — direct `proxy_pass` to a hardcoded hostname would cause nginx to fail at startup if the upstream is down.

Activity logging in nginx writes `/activity/most_recent_call.txt` **only** on `/predict` and `/dataset/` requests, not on static asset fetches. This is intentional: frontend page loads must not keep Fly machines awake.

Both Fly apps use `auto_start_machines = true` + `auto_stop_machines = 'suspend'` (Fly's built-in). The local suspenders provide an additional idle check and send push notifications when machines run too long.

### docker-compose profiles

- **default** (no profile): `backend`, `frontend`, `nginx` — local dev with hot-reload and model bind-mounted from `./checkpoints`
- **production**: `gateway`, `suspender-api`, `suspender-dataset` — no local model, all inference goes to Fly

### Model loading

`api/main.py` loads the most-recently-modified checkpoint from `/checkpoints/`. In dev this is bind-mounted from `./checkpoints/`. In Fly production the checkpoints are baked into the image (`COPY checkpoints/ /checkpoints/` in `api/Dockerfile`). It tries ONNX export via `optimum` first; set `DISABLE_ONNX=true` to force PyTorch.

### Fly deployment contexts

- **API** (`api/fly.toml`): `context: '..'` — deploys from repo root so `checkpoints/` is accessible
- **Dataset** (`dataset/fly.toml`): deploys from `dataset/` — precomputes parquet files at build time in a multi-stage build

### Secrets / manual server setup

Files never synced by Ansible (server-managed):
- `dataset/flyio_token_gateway.txt` — Fly token for dataset suspender
- `dataset/gateway.conf` — WireGuard peer config for dataset app
- `api/flyio_token_gateway_api.txt` — Fly token for API suspender
- `api/gateway.conf` — WireGuard peer config for API app

The gateway container mounts `./dataset/gateway.conf` at runtime. Both gateway WireGuard configs must exist before `gateway-up`.

### Ansible

`ansible/deploy.yml` rsyncs the repo to `/srv/projects/spell` on the server (host `bluh` in `inventory.ini`), then runs `docker compose --profile production up`. **Note:** the playbook still references old service names (`backend`, `dataset-gateway`, `dataset-suspender`, `site`) — needs updating to match current compose (`gateway`, `suspender-api`, `suspender-dataset`).
