.PHONY: start stop check clean logs

COMPOSE_FILE := docker/docker-compose.yml

start:
	docker compose -f $(COMPOSE_FILE) up -d --build
	@echo ""
	@echo "============================================"
	@echo "  JupyterLab est accessible sur :"
	@echo "  http://localhost:8888"
	@echo "============================================"
	@echo ""

stop:
	docker compose -f $(COMPOSE_FILE) down

check:
	docker compose -f $(COMPOSE_FILE) exec mcunet \
		jupyter nbconvert --to notebook --execute \
		/workspace/notebooks/00_setup_check.ipynb \
		--output /tmp/00_setup_check_output.ipynb
	@echo "Setup check OK"

clean:
	docker compose -f $(COMPOSE_FILE) down -v --rmi all --remove-orphans

logs:
	docker compose -f $(COMPOSE_FILE) logs -f mcunet
