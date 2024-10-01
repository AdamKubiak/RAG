#!/bin/bash

# Wait for Ollama to be ready
until $(curl --output /dev/null --silent --head --fail http://ollama:11434); do
    printf '.'
    sleep 5
done

# Pull the required models
curl http://ollama:11434/api/pull -d '{"name":"llama2:3b"}'
curl http://ollama:11434/api/pull -d '{"name":"nomic-embed-text"}'

# Start the Flask application
python app.py