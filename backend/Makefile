.PHONY: reqs

# Variables
PYTHON_VERSION := 3.13
UV_VERSION := 0.8.22

reqs:
	@echo "Generating requirements.txt from requirements.in..."
	docker run --rm \
		-v "$(PWD):/app" \
		-w /app \
		python:$(PYTHON_VERSION)-slim \
		bash -c "pip install uv==$(UV_VERSION) && uv pip compile requirements.in --output-file requirements.txt"
	@echo "✅ requirements.txt generated successfully!"
