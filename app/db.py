from sqlmodel import SQLModel, create_engine, Session

# Create the database engine using the DATABASE_URL from settings
from .settings import DATABASE_URL
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

def get_session():
    with Session(engine) as session:
        yield session