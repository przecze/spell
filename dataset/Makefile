wg-create:
	rm -f gateway.conf
	FLY_API_TOKEN=$$(cat flyio_token_gateway.txt) flyctl wireguard create personal fra dataset-gateway gateway.conf

wg-list:
	FLY_API_TOKEN=$$(cat flyio_token_gateway.txt) flyctl wireguard list

wg-remove-all:
	@export FLY_API_TOKEN=$$(cat flyio_token_gateway.txt) && \
	org=$$(flyctl status -j | jq -r '.Organization.Slug') && \
	for name in $$(flyctl wireguard list -j | jq -r '.[].Name'); do \
		flyctl wireguard remove $$org $$name; \
	done

deploy:
	flyctl deploy

CONFIG_FILES = Dockerfile fly.toml nginx-gateway.conf

copy-config:
	@( for f in $(CONFIG_FILES); do \
		echo "===== $$f ====="; \
		cat $$f; \
		echo; \
	done ) | ( base64 | tr -d '\n' | xargs -0 printf '\033]52;c;%s\a' )
