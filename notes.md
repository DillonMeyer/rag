# TO DO
add alembic

## start api
uvicorn app.main:app --reload

## list docker containers
docker ps

### start docker container
docker compose down

### stop docker container
docker compose down

### wipe DB
docker compose down -v

### enter docker DB container
psql "postgresql://rag:rag@localhost:5432/rag"