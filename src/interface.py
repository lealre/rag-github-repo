import asyncio
import time
from collections.abc import Generator

import streamlit as st

from src.agents.rag_agent import stream_messages


async def stream_response(prompt: str) -> list[str]:
    message_chunks: list[str] = []
    async for chunk in stream_messages(prompt):
        message_chunks.append(chunk)

    return message_chunks


st.title('Chat RAG - Repositório GitHub')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('What is up?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):

        def stream_sync() -> Generator[str, None, None]:
            """
            Transform the AsyncGenerator into a Generator, as Streamlit
            `.write_stream` doesn't support async.
            """
            assert prompt, "Variable 'prompt' is empty or not set!"
            messages = asyncio.run(stream_response(prompt))

            for message in messages:
                yield message
                time.sleep(0.05)

        response = st.write_stream(stream_sync)

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
    })
