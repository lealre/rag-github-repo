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
