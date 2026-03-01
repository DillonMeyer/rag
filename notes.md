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

### run python script module
python -m app.scripts.file_name

### set python source
source .venv/bin/activate

# PostgreSQL

list tables
- psql "postgresql://rag:rag@localhost:5432/rag" -c "\dt"

view a table schema
- psql "postgresql://rag:rag@localhost:5432/rag" -c "\d documents"

view rows
- psql "postgresql://rag:rag@localhost:5432/rag" -c "SELECT document_id, title FROM documents LIMIT 5;"

clean DB
- psql "postgresql://rag:rag@localhost:5432/rag" -c "TRUNCATE chunks, documents RESTART IDENTITY CASCADE;"