import asyncio
import json
import os
import time
from collections import defaultdict

from pydantic_ai import Agent
from pydantic_ai.result import RunResult

from src.agents.contextual_agent import contextual_agent
from src.core.settings import settings
from src.preprocessing.chunk_splitter import (
    num_tokens_from_string,
    save_as_json,
)


async def fetch(agent: Agent[str], chunk: str, key: str) -> RunResult:
    response = await agent.run(chunk, deps=key)
    return response


async def async_fetch(values: list[str], key: str) -> list[RunResult]:
    tasks = [fetch(contextual_agent, value, key) for value in values]
    responses = await asyncio.gather(*tasks)

    return responses


async def generate_context(data: dict[str, list[str]]) -> dict[str, list[str]]:
    json_to_save: dict[str, list[str]] = defaultdict(list[str])
    keys_to_fetch_sync: list[str] = []
    tokens_used: int = 0

    for key in data.keys():
        print('Async process started for', key)

        values = data.get(key)
        assert isinstance(values, list)

        total_tokens = sum([num_tokens_from_string(value) for value in values])

        if total_tokens > settings.MAX_TOKENS_PER_MINUITE:
            print(
                f'Async process skipped for {key} due to the number of tokens.'
            )
            keys_to_fetch_sync.append(key)
            continue

        if (total_tokens + tokens_used) > settings.MAX_TOKENS_PER_MINUITE:
            tokens_used = 0
            print('Waiting 60 seconds due to TPM limit...')
            time.sleep(60)

        responses = await async_fetch(values, key)

        response_tokens = sum([
            response.usage().total_tokens or 0 for response in responses
        ])

        tokens_used += response_tokens

        chunk_context = [response.data for response in responses]
        json_to_save[key] = chunk_context

    for key in keys_to_fetch_sync:
        print('Sync process started for', key)
        values = data.get(key)
        assert isinstance(values, list)

        aggregated_contexts: list[str] = []

        for n, value in enumerate(values):
            total_tokens = num_tokens_from_string(value)

            if (total_tokens + tokens_used) > settings.MAX_TOKENS_PER_MINUITE:
                tokens_used = 0
                print('Waiting 60 seconds due to TPM limit...')
                time.sleep(60)

            response = await contextual_agent.run(value, deps=key)

            response_tokens = response.usage().total_tokens or 0
            tokens_used += response_tokens

            aggregated_contexts.append(response.data)

            print(f'{n}/{len(values)}')

        json_to_save[key] = aggregated_contexts

    return json_to_save


if __name__ == '__main__':
    folder = 'data'
    chunked_data = 'data_chunks.json'
    file_name = 'final_data.json'
    input_path = os.path.join(folder, chunked_data)

    with open(input_path, 'r', encoding='utf-8') as json_file:
        data_chunks: dict[str, list[str]] = json.load(json_file)

    contextual_data = asyncio.run(generate_context(data_chunks))

    final_data: dict[str, list[str]] = {}
    for key, _ in data_chunks.items():
        final_data[key] = [
            f'{x}\n{y}' for x, y in zip(contextual_data[key], data_chunks[key])
        ]

    save_as_json(data=final_data, output_dir=folder, file_name=file_name)
