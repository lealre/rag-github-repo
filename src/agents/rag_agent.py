"""
RAG example from LangChain using pgvector as database

Documents from the repository of Jornada de Dados
"""

import asyncio
import sys
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import asyncpg
import pydantic_core
from openai import AsyncOpenAI
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

from src.core.database import database_connect


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


system_prompt = """
Você é um agente da Jornada de Dados, uma comunidade que ensina projetos
práticos de engenharia de dados. Sua função é responder sobre o material
que está no repositório da jornada. Você irá receber o noma da pasta raiz e
parte do conteudo que esta nela. Sobras as pastas:
- Se tiver algum numero no nomae da pasta, se trata to númerpo do Workshops
que foi ministrado.
- As que tivrerem Bootcamp são de Bootcamps
- O resto são aulas normais.

Use a função `retrieve` para buscra resultados a respeito da pergunta do
usuário
"""

agent = Agent(
    'openai:gpt-3.5-turbo',
    system_prompt=system_prompt,
    deps_type=Deps,
    result_type=str,
)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """

    response = await context.deps.openai.embeddings.create(
        input=search_query,
        model='text-embedding-3-small',
    )

    embedding = response.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        """
        SELECT folder, content FROM repo
        ORDER BY embedding <=> $1 LIMIT 1
        """,
        embedding_json,
    )

    return '\n\n'.join(f'Conteudo:\n{row["content"]}\n' for row in rows)


async def stream_messages(question: str) -> AsyncGenerator[str, None]:
    """
    Stream messages for Streamlit interface.
    """
    openai = AsyncOpenAI()

    async with database_connect() as pool:
        deps = Deps(openai=openai, pool=pool)
        async with agent.run_stream(question, deps=deps) as result:
            async for message in result.stream_text(delta=True):
                yield message


async def run_agent(question: str) -> None:
    """
    Entry point to run the agent and perform RAG based question answering.
    """
    openai = AsyncOpenAI()

    async with database_connect() as pool:
        deps = Deps(openai=openai, pool=pool)
        answer = await agent.run(question, deps=deps)

    print(answer.data)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None

    if action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'Qual problema é resolvido no workshop de kafka?'
        asyncio.run(run_agent(q))
    else:
        print('Exiting')
