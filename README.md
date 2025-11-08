## Mem0 Small Model Integration

A research project integrating Mem0's memory layer with a fine-tuned Llama 3.1 8B Instruct model for memory-driven conversational AI on limited hardware.

## Overview

This project implements a complete pipeline for fine-tuning small LLMs and integrating them with Mem0's hybrid memory system. The system enables personalized AI conversations that remember context across sessions while running efficiently on consumer hardware.

## Project Structure

```plaintext
.
├── finetune_scripts/
│   └── Unsloth_fine_tuning_lambda_cloud.ipynb  # Fine-tuning notebook
├── mem0_backend/
│   ├── mem0_config.py                      # Mem0 configuration
│   └── cli_chat.py                         # CLI chat interface
├── benchmarks/
│   ├── run_benchmark_without_memo.py                 # Baseline inference benchmark
│   └── run_benchmark_retrievel.py        # Memory retrieval evaluation
├── data_download/
│   ├── coqa_mctest_5000_samples.jsonl            # Training dataset for fine-tuning
│   ├── benchmark_prompts_latency.jsonl          # Inference test prompts
│   └── benchmark_test_retreivel_quality.jsonl  # Retrieval test data
└── README.md
````

## System Requirements

### Hardware

  * **Minimum:** 16GB RAM, CPU-only inference with Ollama
  * **Recommended:** NVIDIA GPU with 24GB+ VRAM for fine-tuning
  * **Cloud:** Lambda Cloud A100 instances (used in this project for fine-tuning)

### Software

  * Python 3.10+
  * CUDA 12.8+ (for GPU acceleration)
  * Ollama (local inference)
  * Qdrant (vector database)

## Installation

### 1\. Core Dependencies

```bash
# Install Python packages
pip install mem0ai ollama-python unsloth datasets trl transformers torch
# Install Ollama (macOS/Linux)
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

# Start Ollama service
ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M (in another terminal)
ollama pull mxbai-embed-large:latest (in another terminal)
```

### 2\. Qdrant Vector Database

```bash
# Install locally (see [https://qdrant.tech/documentation/quick-start/](https://qdrant.tech/documentation/quick-start/))
```

### 3\. Clone and Setup

```bash
git clone <your-repo-url>
cd Mem0_finetuned_llama_backend
```

## Fine-Tuning Pipeline

### Dataset Preparation

The project uses a 5,000-sample conversational dataset combining CoQA and MCTest data formatted for instruction fine-tuning. Each sample follows the Llama 3.1 chat template format:

```json
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant with excellent long-term conversational memory. Use the conversation history to answer questions accurately with specific details.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n**Conversation History:**\n[This is a memory from your past.]\nMy dad runs the Blue Street Zoo. Everyone calls him the Zoo King. That means Mom is the Zoo Queen. And that means that I'm the Zoo Prince! Being a prince is very special. \n\nI spend every morning walking around to see the zoo. It's better than any animal book. I say hello to the lions. I say woof at all of the wolves. I make faces to the penguins. Once I even gave a morning kiss to a bear! My favorite animal is the piggy. I named him Samson. He likes to eat mustard, so I toss some mustard jars into his cage every morning. I don't know why that piggy likes mustard so much. \n\nSometimes I walk around with the Zoo King and Zoo Queen. Then we say hello to the animals together! I really like those days. Everybody who works at the Zoo says hello to us when we walk by. At lunchtime, we all go to the Zoo restaurant and eat pork chops. I hope Samson doesn't get mad about that!\n\nWho has been kissed?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\na bear<|eot_id|><|end_of_text|>", "question": "Who has been kissed?", "answer": "a bear", "source": "mctest"}
```

### Training Configuration

  * **Model:** Llama 3.1 8B Instruct (unsloth/Meta-Llama-3.1-8B-Instruct)
  * **LoRA Parameters:**
      * Rank (r): 16
      * Alpha: 16
      * Dropout: 0
      * Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  * **Training Hyperparameters:**
      * Per-device batch size: 4
      * Gradient accumulation: 4 (effective batch size: 16)
      * Learning rate: 2e-4
      * Max sequence length: 2048
      * Optimizer: AdamW 8-bit
      * Scheduler: Linear warmup (5 steps) + decay
      * Training steps: 562 (2 epochs over 4,500 samples)
      * Evaluation: Every 100 steps
      * Checkpointing: Best model saved based on validation loss

### Running Fine-Tuning

Open the Jupyter notebook:

```bash
jupyter notebook Unsloth_fine_tuning_lambda_cloud.ipynb
```

The training process achieves:

  * **Initial loss:** 1.297
  * **Final validation loss:** 0.077 (94% reduction)
  * **Training time:** \~30-40 minutes on A100 GPU

### Model Export

After training, export to GGUF format for Ollama:

```python
model.save_pretrained_gguf("fine_tuned_model", tokenizer, quantization_method="q4_k_m")
```
Place the gguf in mem0_backend folder 

Load into Ollama:

```bash
cd mem0_backend
ollama create fine_tuned:latest -f Modelfile
```

## Mem0 Configuration

### Backend Setup (mem0_config.py)

```python
CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "fine_tuned:latest", or "llama3.1:8b-instruct-q4_K_M"
            "temperature": 0.1,
            "top_p": 0.1,
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
            "path": "./qdrant_chat_db_cli"
        }
    },
    "version": "v1.0"
}
```

This configuration uses:

  * **LLM:** Your fine-tuned model or baseline model via Ollama for memory extraction and chat
  * **Embedder:** MXBai-large (1024-dim) for semantic search
  * **Vector Store:** Local Qdrant for memory persistence

## Usage

### Interactive Chat Interface

```bash
python mem0_backend/cli_chat.py
```

**Features:**

  * Automatic memory search before each response
  * Conversation auto-save to long-term memory
  * Manual memory management commands

**Commands:**

```text
You: Hello, my name is Varun and I work at XYZ

🤖 Nice to meet you, Varun! It's great to connect with someone from XYZ...
💭 Used 0 memories
✓ Conversation saved to memory

You: /search work
📚 Found 1 memories:
1. User asked: Hello, my name is Varun and I work at XYZ
   Assistant replied: Nice to meet you...
   Score: 0.847

You: What's my name?
📚 Found 1 memories (searching context)
🤖 Your name is Varun.
💭 Used 1 memories
✓ Conversation saved to memory
```

## Benchmarking

### 1\. Baseline Inference Performance

Tests raw model latency and throughput without memory retrieval:

```bash
python benchmark_baseline.py
```

**Measured Metrics:**

  * Latency (ms): min, max, mean, median, P90, P95, stdev
  * Throughput: tokens/sec, prompts/sec
  * Token statistics: total tokens, avg per prompt
  * Output: `benchmark_results_without_memo_baseline.json`

**Results for llama3.1:8b-instruct-q4_K_M":**

```json
{"model": "llama3.1:8b-instruct-q4_K_M",
    "backend": "Ollama (Direct Inference)",
    "total_prompts": 85,
    "successful": 85,
    "failed": 0,
    "inference_latency_ms": {
      "min": 1138.685941696167,
      "max": 26432.260990142822,
      "mean": 4436.906259200153,
      "median": 3579.1895389556885,
      "stdev": 3428.128815605243,
      "p90": 6735.480070114136,
      "p95": 7500.785827636719
    },
    "throughput": {
      "tokens_per_second": 9.442191292680395,
      "prompts_per_second": 0.22538226899124786
    },
    "token_stats": {
      "total_tokens": 3561,
      "avg_tokens_per_prompt": 41.89411764705882
    }}
```

**Results for our fine-tuned model":**

```json
{"model": "fine_tuned:latest",
    "backend": "Ollama (Direct Inference)",
    "total_prompts": 85,
    "successful": 85,
    "failed": 0,
    "inference_latency_ms": {
      "min": 445.6310272216797,
      "max": 4246.412992477417,
      "mean": 797.5396016064811,
      "median": 563.7807846069336,
      "stdev": 660.8583660364415,
      "p90": 1022.7198600769043,
      "p95": 2733.7210178375244
    },
    "throughput": {
      "tokens_per_second": 4.690897433873373,
      "prompts_per_second": 1.2538562323246438
    },
    "token_stats": {
      "total_tokens": 318,
      "avg_tokens_per_prompt": 3.7411764705882353
    }}
```

### 2\. Memory Retrieval Quality

Evaluates Mem0's semantic search accuracy:

```bash
python benchmark_retrieval_quality.py
```

## Key Features

### 1\. Efficient Fine-Tuning

  * Gradient checkpointing enables training on limited hardware
  * Fast training: 94% loss reduction in 562 steps

### 2\. Hybrid Memory Architecture

  * Vector search for semantic similarity
  * Metadata filtering with ground-truth IDs
  * Automatic deduplication and memory consolidation

### 3\. Production-Ready Pipeline

  * Error handling for failed searches
  * Detailed logging and metrics tracking
  * Progress monitoring during batch operations
  * Reproducible benchmark results
