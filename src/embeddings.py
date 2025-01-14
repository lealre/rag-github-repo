"""
This script populates the database with the vector embeddings
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import pydantic_core
from openai import AsyncOpenAI

from src.core.database import database_connect


@dataclass
class Record:
    folder: str
    content: str


async def populate_db(data: dict[str, list[str]]) -> None:
    openai = AsyncOpenAI()

    async with database_connect() as pool:
        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for key, values in data.items():
                for value in values:
                    record = Record(folder=key, content=value)
                    tg.create_task(insert_record(sem, openai, pool, record))


async def insert_record(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    record: Record,
) -> None:
    async with sem:
        print(f'Populating {record.folder}')
        embedding = await openai.embeddings.create(
            input=record.content,
            model='text-embedding-3-small',
        )

        embedding = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            """
            INSERT INTO repo (folder, content, embedding)
            VALUES ($1, $2, $3)
            """,
            record.folder,
            record.content,
            embedding_json,
        )


if __name__ == '__main__':

    folder = 'data'
    file = 'final_data.json'
    file_path = os.path.join(folder, file)

    with open(file_path, 'r', encoding='utf-8') as file:
        data: dict[str, list[str]] = json.load(file)

    asyncio.run(populate_db(data))