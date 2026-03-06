import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://rag:rag@db:5432/rag"
)

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://ollama:11434"
)

DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL",
    "llama3.2:latest"
)