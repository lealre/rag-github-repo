#!/bin/sh

# Create tables in the database
uv run -m src.core.database

# Generate embeddings
uv run -m src.embeddings

# Start the application
uv run -m streamlit run src/interface.py
