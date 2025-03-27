up:
	docker-compose up --build

down:
	docker-compose down

logs:
	docker-compose logs -f

restart:
	docker-compose down && docker-compose up --build

api:
	docker-compose exec api bash

bot:
	docker-compose exec bot bash

streamlit:
	docker-compose exec streamlit bash

psql:
	docker-compose exec db psql -U postgres -d mydb

help:
	@echo "🛠️  Commandes disponibles :"
	@echo "  make up         ⟶ Démarrer tous les services"
	@echo "  make down       ⟶ Arrêter tous les services"
	@echo "  make logs       ⟶ Afficher les logs en temps réel"
	@echo "  make restart    ⟶ Redémarrer tous les services"
	@echo "  make api        ⟶ Accéder au conteneur API"
	@echo "  make bot        ⟶ Accéder au conteneur Bot"
	@echo "  make streamlit  ⟶ Accéder au conteneur Streamlit"
	@echo "  make psql       ⟶ Ouvrir psql dans le conteneur DB"
