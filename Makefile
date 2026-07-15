.PHONY: start shell run check stop clean logs

COMPOSE_FILE := docker/docker-compose.yml
EX ?= 00

start:
	docker compose -f $(COMPOSE_FILE) up -d --build
	@echo ""
	@echo "============================================================"
	@echo "  Container pret."
	@echo "  Se connecter avec :   make shell"
	@echo "  Puis lancer :         python exercises/00_setup_check.py"
	@echo "============================================================"
	@echo ""

# Ouvre un shell interactif dans le container (affiche le guide START_HERE).
shell:
	docker compose -f $(COMPOSE_FILE) exec mcunet bash

# Lance un exercice sans entrer dans le shell.  Ex :  make run EX=02
run:
	docker compose -f $(COMPOSE_FILE) exec mcunet bash -lc 'python exercises/$(EX)_*.py'

# Verifie l'environnement en executant l'exercice 00.
check:
	docker compose -f $(COMPOSE_FILE) exec mcunet python exercises/00_setup_check.py
	@echo "Setup check OK"

stop:
	docker compose -f $(COMPOSE_FILE) down

clean:
	docker compose -f $(COMPOSE_FILE) down -v --rmi all --remove-orphans

logs:
	docker compose -f $(COMPOSE_FILE) logs -f mcunet
