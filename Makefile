.PHONY: deploy deploy-check fly-deploy fly-deploy-api fly-deploy-dataset fly-api-create gateway-up gateway-down api-wg-create dataset-wg-create dataset-wg-list dataset-wg-remove-all api-reqs print-infra-files dump-infra copy-infra

# ── Deployment ───────────────────────────────────────────────────────────────

deploy:
	cd ansible && ansible-playbook deploy.yml

deploy-check:
	cd ansible && ansible-playbook deploy.yml --check --diff

# ── Fly.io ───────────────────────────────────────────────────────────────────

fly-deploy: fly-deploy-api fly-deploy-dataset

fly-deploy-api:
	flyctl deploy --config api/fly.toml

fly-deploy-dataset:
	flyctl deploy --config dataset/fly.toml

# Run once to create the API Fly app before the first deploy
fly-api-create:
	flyctl apps create api-spell-jan-czechowski-com
	flyctl ips allocate-v6 --private -a api-spell-jan-czechowski-com

# ── Local gateway (Fly.io backend, localhost frontend) ───────────────────────
# Run the production gateway locally on http://localhost:3080
# Frontend is baked into the image; /predict and /dataset/ go to Fly.io via WireGuard.

gateway-up:
	docker network inspect nginx-proxy >/dev/null 2>&1 || docker network create nginx-proxy
	docker compose --profile production up --build gateway suspender-api suspender-dataset

gateway-down:
	docker compose --profile production down gateway suspender-api suspender-dataset

api-wg-create:
	rm -f api/gateway.conf
	FLY_API_TOKEN=$$(cat api/flyio_token_gateway_api.txt) flyctl wireguard create personal fra api-gateway api/gateway.conf

dataset-wg-create:
	rm -f dataset/gateway.conf
	FLY_API_TOKEN=$$(cat dataset/flyio_token_gateway.txt) flyctl wireguard create personal fra dataset-gateway dataset/gateway.conf

dataset-wg-list:
	FLY_API_TOKEN=$$(cat dataset/flyio_token_gateway.txt) flyctl wireguard list

dataset-wg-remove-all:
	@export FLY_API_TOKEN=$$(cat dataset/flyio_token_gateway.txt) && \
	org=$$(flyctl status -j | jq -r '.Organization.Slug') && \
	for name in $$(flyctl wireguard list -j | jq -r '.[].Name'); do \
		flyctl wireguard remove $$org $$name; \
	done

# ── Backend ──────────────────────────────────────────────────────────────────

PYTHON_VERSION := 3.13
UV_VERSION := 0.8.22

# Always CPU-only — used for both local dev and Fly deployment
api-reqs:
	@echo "Compiling api/requirements.txt (CPU-only torch)..."
	docker run --rm \
		-v "$(PWD)/api:/app" \
		-w /app \
		python:$(PYTHON_VERSION)-slim \
		bash -c "pip install uv==$(UV_VERSION) && uv pip compile requirements.in \
			--index-url https://download.pytorch.org/whl/cpu \
			--extra-index-url https://pypi.org/simple \
			--output-file requirements.txt"
	@echo "Verifying no CUDA packages..."
	@grep -iE 'nvidia|cuda|cu[0-9]{2,3}' api/requirements.txt && \
		{ echo "ERROR: CUDA packages found in requirements.txt"; exit 1; } || \
		echo "OK: no CUDA packages found"

# ── Infra dump ───────────────────────────────────────────────────────────────

INFRA_FILES := \
	Makefile \
	docker-compose.yml \
	ansible/deploy.yml \
	api/fly.toml \
	api/Dockerfile \
	dataset/fly.toml \
	dataset/Dockerfile.fly \
	Dockerfile.gateway \
	gateway/nginx.conf \
	gateway/entrypoint.sh \
	nginx.conf.template \
	Dockerfile.frontend \
	api/Dockerfile \
	api/Dockerfile.dev \
	dataset/suspender.sh \
	dataset/notify.sh

print-infra-files:
	@printf '%s\n' $(INFRA_FILES)

dump-infra:
	@set -eu; \
	for f in $(INFRA_FILES); do \
		printf '===== %s =====\n' "$$f"; \
		if [ -f "$$f" ]; then \
			cat "$$f"; \
		else \
			printf '[missing]\n'; \
		fi; \
		printf '\n'; \
	done

copy-infra:
	@set -eu; \
	( \
		printf '===== INFRA_FILES =====\n'; \
		for f in $(INFRA_FILES); do printf '%s\n' "$$f"; done; \
		printf '\n'; \
		$(MAKE) --no-print-directory dump-infra; \
	) | base64 | tr -d '\n' | awk '{ printf "\033]52;c;%s\a", $$0 }'
	@printf 'copied infra dump to OSC52 clipboard\n'
