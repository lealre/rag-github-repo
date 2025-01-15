# RAG App to Retrieve Information from a GitHub Repository

This repository consists of a simple RAG application to retrieve information about a GitHub repository, specifically [this one](https://github.com/lvgalvao/data-engineering-roadmap).

It uses:

- [PydanticAI](https://docs.pydantic.dev/latest/) as the agent framework.
- PostgreSQL as the vector database, relying on [`pgvector`](https://github.com/pgvector/pgvector) and [`asyncpg`](https://magicstack.github.io/asyncpg/current/) libraries.
- [OpenAI API](https://platform.openai.com/docs/overview) to generate the embeddings.
- [Streamlit](https://streamlit.io/) to build the chat interface.

Below is a quick demonstration of the chat running in Docker Compose, along with the PostgreSQL database. It queries files in two specific directories from the repository.

![](media/rag-demo.gif)

## How it works

The data was extracted using [git ingest](https://gitingest.com/), which provides all the content of a repository in a TXT file.

To use as embeddings, the data was split into chunks of a maximum of 6000 tokens, while always maintaining the same root folder context. No overlap between chunks was applied.

In an attempt to give more context to the data, a preprocessing stage was done, where a preprocessing agent was created. Its function was to append a brief context describing the content of each chunk.

The cosine distance method was used to compare embeddings and retrieve relevant information.

Although it serves as a simple way to implement a RAG using these tools, the overall performance of the RAG application in simple usage was poor, with constant hallucinations if general questions about the repository are asked. One of the possible causes is the chunk size of the files, as some words tend to give more weight to specific files when searching in the database, even with different combinations of other words.

The process used for embedding and extracting the vectors was based on the PydanticAI [RAG documentation example](https://ai.pydantic.dev/examples/rag/), and the chat does not pass the message history to the model.

## How to Run This Project

This section shows how to run the project using Docker by building the two services together: the Chat interface and the PostgreSQL database.

The Docker Compose setup includes an `entrypoint.sh` file, which runs the Python scripts responsible for creating the vector database. Be aware that it uses the OpenAI API key, which must be set in the `.env` file.

[How to install Docker Compose](https://docs.docker.com/compose/install/)

1 - Clone the repo locally:

```shell
git clone https://github.com/lealre/github-repo-agent.git
```

2 - Access the project directory:

```shell
cd github-repo-agent
```

3 - Create the `.env` file by renaming the `.env-example`:

```shell
mv .env-example .env
```

Insert your API key in `OPENAI_API_KEY`.

4 - Build and start the container:

```shell
docker compose up
```

The application will be available at `http://localhost:8501/`.
