import asyncio
import time

import streamlit as st

from src.agents.rag_agent import stream_messages

st.title('Agente RAG - Repositório GitHub')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('What is up?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    async def stream_response():
        message_chunks: list = []
        assert prompt
        async for chunk in stream_messages(prompt):
            message_chunks.append(chunk)

        return message_chunks

    def stream_sync():
        messages = asyncio.run(stream_response())
        for message in messages:
            yield message
            time.sleep(0.05)

    with st.chat_message('assistant'):
        response = st.write_stream(stream_sync)

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
    })
