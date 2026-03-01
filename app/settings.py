import os

# fallback to a default database URL if not set in environment variables
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://rag:rag@localhost:5432/rag"
)