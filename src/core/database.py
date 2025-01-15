import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg

from src.core.settings import settings


@asynccontextmanager
async def database_connect() -> AsyncGenerator[asyncpg.Pool, None]:
    try:
        pool = await asyncpg.create_pool(settings.DATABASE_URL)
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repo (
    id serial PRIMARY KEY,
    folder text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);

CREATE INDEX
IF NOT EXISTS idx_repo_embedding
ON repo
USING hnsw (embedding vector_l2_ops);
"""


async def build_search_db() -> None:
    async with database_connect() as pool:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(DB_SCHEMA)


if __name__ == '__main__':
    asyncio.run(build_search_db())
