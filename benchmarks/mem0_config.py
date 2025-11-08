CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "fine_tuned:latest", # or "llama3.1:8b-instruct-q4_K_M"
            "temperature": 0.7,
            "top_p": 0.9,
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large:latest",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_chat",
            "embedding_model_dims": 1024,
            "path": "./qdrant_chat_db_ret"
        }
    },
    "version": "v1.0" 
}
