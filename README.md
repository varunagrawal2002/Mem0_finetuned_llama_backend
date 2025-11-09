## Mem0 Small Model Integration

A project integrating Mem0's memory layer with a fine-tuned Llama 3.1 8B Instruct model for memory-driven conversational AI on limited hardware.

## Overview

This project implements a complete pipeline for fine-tuning small LLMs like Llama 3.1 8B Instruct and integrating them with Mem0's hybrid memory system. The system enables personalized AI conversations that remember context across sessions while running efficiently on consumer hardware.

## Project Structure

```plaintext
.
├── finetune_scripts/
│   └── Unsloth_fine_tuning_lambda_cloud.ipynb  # Fine-tuning notebook (earlier)
│   └── Unsloth_fine_tuning_locomo_lambda_cloud.ipynb  # Fine-tuning notebook (latest)
├── mem0_backend/
│   ├── mem0_config.py                      # Mem0 configuration
│   └── cli_chat.py                         # CLI chat interface
├── benchmarks/
│   ├── run_benchmark_without_memo.py                 # Baseline inference benchmark
│   └── run_benchmark_retrievel.py        # Memory retrieval evaluation
├── data_download/
│   ├── coqa_mctest_5000_samples.jsonl            # Training dataset for fine-tuning (earlier)
│   ├── benchmark_prompts_latency.jsonl          # Inference test prompts
│   ├── mem0_finetune_locomo_dataset.jsonl          # Training dataset for fine-tuning (latest)
├── benchmark_test_retreivel_quality.jsonl  # Retrieval test data
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

This iteration uses the LoCoMo dataset instead of previous datasets. A sample of this dataset is attached in the latest benchmark report.

The prior CoQA 5,000-sample conversational dataset is replaced with this to better align with Mem0's hybrid memory setup.

```json
# LOCOMO Dataset example (latest)

{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert memory extraction assistant. Your job is to read the following short conversation and extract any new, important facts about the speakers' lives, preferences, or events as a concise, single-line summary. If no important fact is present, output 'None'.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMelanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?\nCaroline: I went to a LGBTQ support group yesterday and it was so powerful.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nCaroline attended an LGBTQ support group recently and found the transgender stories inspiring.<|eot_id|><|end_of_text|>"}
```
```json
# CoQA Dataset example (earlier)

{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant with excellent long-term conversational memory. Use the conversation history to answer questions accurately with specific details.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n**Conversation History:**\n[This is a memory from your past.]\nMy dad runs the Blue Street Zoo. Everyone calls him the Zoo King. That means Mom is the Zoo Queen. And that means that I'm the Zoo Prince! Being a prince is very special. \n\nI spend every morning walking around to see the zoo. It's better than any animal book. I say hello to the lions. I say woof at all of the wolves. I make faces to the penguins. Once I even gave a morning kiss to a bear! My favorite animal is the piggy. I named him Samson. He likes to eat mustard, so I toss some mustard jars into his cage every morning. I don't know why that piggy likes mustard so much. \n\nSometimes I walk around with the Zoo King and Zoo Queen. Then we say hello to the animals together! I really like those days. Everybody who works at the Zoo says hello to us when we walk by. At lunchtime, we all go to the Zoo restaurant and eat pork chops. I hope Samson doesn't get mad about that!\n\nWho has been kissed?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\na bear<|eot_id|><|end_of_text|>", "question": "Who has been kissed?", "answer": "a bear", "source": "mctest"}
```

### Training Configuration

  * **Model:** Llama 3.1 8B Instruct (unsloth/Meta-Llama-3.1-8B-Instruct)
  * **LoRA Parameters:**
      * Rank (r): 12
      * Alpha: 12
      * Dropout: 0.1
      * Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  * **Training Hyperparameters:**
      * Per-device batch size: 4
      * Gradient accumulation: 4 (effective batch size: 16)
      * Learning rate: 2e-4
      * Max sequence length: 2048
      * Optimizer: AdamW 8-bit
      * Scheduler: Linear warmup (5 steps) + decay
      * Training steps: 250
      * Evaluation: Every 20 steps
      * Checkpointing: Best model saved based on validation loss

### Running Fine-Tuning

Open the Jupyter notebook:

```bash
jupyter notebook Unsloth_fine_tuning_locomo_lambda_cloud.ipynb
```

The training process achieves:

  * **Initial loss:** 1.29
  * **Final validation loss:** 0.90 
  * **Training time:** \~5-10 minutes on A100 GPU

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

### Backend Setup (mem0_backend/mem0_config.py)

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
            "path": "./qdrant_chat_db"
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

You: What's my name?
📚 Found 1 memories:
🤖 Your name is Varun.
💭 Used 1 memories
✓ Conversation saved to memory
```

## Benchmarking

### 1. Baseline Inference Performance

Tests raw model latency and throughput without memory retrieval (set the model in `benchmarks/mem0_config.py`):

```bash
run_benchmark_without_memo.py
```

**Measured Metrics:**

* Latency (s): min, max, mean, median, P90, P95, stdev
* Throughput: tokens/sec, prompts/sec
* Token statistics: total tokens, avg per prompt
* Output: `benchmarks/benchmark_results_without_memo_baseline.json` and `benchmarks/benchmark_results_without_memo_finetuned.json` (earlier benchmarks)
* Output: `benchmarks/benchmark_results_without_memo_baseline.json` and `benchmarks/benchmark_results_without_memo_finetuned_locomo.json` (latest benchmarks)

---

**Results for `llama3.1:8b-instruct-q4_K_M`:**

```json
{
  "model": "llama3.1:8b-instruct-q4_K_M",
  "backend": "Ollama (Direct Inference)",
  "total_prompts": 85,
  "successful": 85,
  "failed": 0,
  "inference_latency_s": {
    "min": 1.139,
    "max": 26.432,
    "mean": 4.437,
    "median": 3.579,
    "stdev": 3.428,
    "p90": 6.735,
    "p95": 7.501
  },
  "throughput": {
    "tokens_per_second": 9.442,
    "prompts_per_second": 0.225
  },
  "token_stats": {
    "total_tokens": 3561,
    "avg_tokens_per_prompt": 41.894
  }
}
```

---

**Results for our fine-tuned model (latest fine-tuning):**

```json
{
  "model": "fine_tuned:latest",
  "backend": "Ollama (Direct Inference)",
  "total_prompts": 85,
  "successful": 85,
  "failed": 0,
  "inference_latency_s": {
    "min": 0.901,
    "max": 11.090,
    "mean": 3.193,
    "median": 2.623,
    "stdev": 1.800,
    "p90": 5.908,
    "p95": 6.417
  },
  "throughput": {
    "tokens_per_second": 9.267,
    "prompts_per_second": 0.313
  },
  "token_stats": {
    "total_tokens": 2515,
    "avg_tokens_per_prompt": 29.588
  }
}
```

---

**Results for our fine-tuned model (earlier fine-tuning):**

```json
{
  "model": "fine_tuned:latest",
  "backend": "Ollama (Direct Inference)",
  "total_prompts": 85,
  "successful": 85,
  "failed": 0,
  "inference_latency_s": {
    "min": 0.446,
    "max": 4.246,
    "mean": 0.798,
    "median": 0.564,
    "stdev": 0.661,
    "p90": 1.023,
    "p95": 2.734
  },
  "throughput": {
    "tokens_per_second": 4.691,
    "prompts_per_second": 1.254
  },
  "token_stats": {
    "total_tokens": 318,
    "avg_tokens_per_prompt": 3.741
  }
}
```

### 2\. Memory Retrieval Quality

Evaluates Mem0's semantic search accuracy (set the model in benchmarks/mem0_config.py):

**Measured Metrics:**

The benchmark evaluates retrieval quality by checking if the correct memory (`ground_truth_id`) is returned within the top 5 search results.

* **Hits@k (or Recall@k):** This shows how often the correct memory was found *within the top 'k' results*.
    * **Hits@1 (18.0%):** The correct memory was the **#1 top-ranked result** 18% of the time. This is the most important "hit" metric.
    * **Hits@5 (24.0%):** The correct memory was found **anywhere in the top 5 results** 24% of the time.

* **Precision@5 (4.8%):** Of all the items retrieved (100 queries * 5 results = 500 total), this is the percentage that were the correct one. It's calculated as `Total Hits@5 / (Total Queries * 5)`.

* **Mean Reciprocal Rank (MRR) (0.2037):** This is the most comprehensive metric for ranking quality. It heavily rewards results at the top of the list. It is the average "reciprocal rank" score across all queries:
    * Correct at Rank 1 = **1.0** points
    * Correct at Rank 2 = **0.5** points (1/2)
    * Correct at Rank 3 = **0.33** points (1/3) ....
    * Not found in top 5 = **0** points
    * A perfect score (always #1) is **1.0**.

Output: `benchmark_results_baseline(or finetuned).json`

```bash
python run_benchmark_retrievel.py
```
---

**Results for llama3.1:8b-instruct-q4_K_M":**

```json
{
    "total_queries": 100,
    "successful_queries": 100,
    "failed_queries": 0,
    "hits_at_1": 25,
    "hits_at_3": 29,
    "hits_at_5": 32,
    "recall_at_1_pct": 25.0,
    "recall_at_3_pct": 29.0,
    "recall_at_5_pct": 32.0,
    "precision_at_5_pct": 6.4,
    "mean_reciprocal_rank": 0.277
  }
```
---

**Results for our fine-tuned model":**

```json

# Latest fine-tuning
{
    "total_queries": 100,
    "successful_queries": 100,
    "failed_queries": 0,
    "hits_at_1": 53,
    "hits_at_3": 56,
    "hits_at_5": 57,
    "recall_at_1_pct": 53.0,
    "recall_at_3_pct": 56.0,
    "recall_at_5_pct": 57.0,
    "precision_at_5_pct": 11.4,
    "mean_reciprocal_rank": 0.547
  }
```
---

```json

# Earlier fine-tuning
{
    "total_queries": 100,
    "successful_queries": 100,
    "failed_queries": 0,
    "hits_at_1": 18,
    "hits_at_3": 23,
    "hits_at_5": 24,
    "recall_at_1_pct": 18.0,
    "recall_at_3_pct": 23.0,
    "recall_at_5_pct": 24.0,
    "precision_at_5_pct": 4.8,
    "mean_reciprocal_rank": 0.2037
  }
```

## Key Features

### 1\. Efficient Fine-Tuning

  * Gradient checkpointing enables training on limited hardware

### 2\. Hybrid Memory Architecture

  * Vector search for semantic similarity
  * Metadata filtering with ground-truth IDs
  * Automatic deduplication and memory consolidation

### 3\. Production-Ready Pipeline

  * Error handling for failed searches
  * Detailed logging and metrics tracking
  * Progress monitoring during batch operations
  * Reproducible benchmark results
