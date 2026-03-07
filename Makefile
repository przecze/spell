.PHONY: deploy down logs build
# Deploy to production server
deploy:
	cd ansible && ansible-playbook deploy.yml

# Deploy with dry-run (shows what would happen)
deploy-check:
	cd ansible && ansible-playbook deploy.yml --check --diff


# root + fly infra/config dump
# includes:
# - compose files
# - nginx configs
# - fly.toml / gateway / entrypoint / suspender
# - Dockerfiles / Makefiles
# - duplicated-ish requirements and dataset runtime files for diffing
# - deployment notes

INFRA_FILES := \
	Makefile \
	README.md \
	docker-compose.yml \
	nginx.conf \
	nginx.conf.template \
	Dockerfile.frontend \
	frontend/Dockerfile \
	backend/Dockerfile \
	backend/Makefile \
	dataset/Dockerfile \
	dataset/Dockerfile \
	dataset/Makefile \
	dataset/docker-compose.yml \
	dataset/entrypoint.gateway.sh \
	dataset/fly.toml \
	dataset/gateway.conf \
	dataset/nginx-gateway.conf \
	dataset/notify.sh \
	dataset/precompute.py \
	dataset/requirements-runtime.in \
	dataset/requirements-runtime.txt \
	dataset/requirements.txt \
	dataset/suspender.sh

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

