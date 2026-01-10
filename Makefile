.PHONY: deploy down logs build
# Deploy to production server
deploy:
	cd ansible && ansible-playbook deploy.yml

# Deploy with dry-run (shows what would happen)
deploy-check:
	cd ansible && ansible-playbook deploy.yml --check --diff



