import json
import time
from mem0 import Memory

JSONL_FILE = "benchmark_test_retreivel_quality.jsonl"

print(f"Loading test dataset from '{JSONL_FILE}'...")

SYNTHETIC_DATASET = []

try:
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.strip()

    if not content.startswith('['):
        content = '[' + content.rstrip(',') + ']'

    try:
        SYNTHETIC_DATASET = json.loads(content)
        print(f"✓ Loaded {len(SYNTHETIC_DATASET)} test examples from {JSONL_FILE}")
    except json.JSONDecodeError as e:
        print(f"Error parsing as JSON array: {e}")
        print("\nTrying alternative parsing method...")

        objects = []
        stack = []
        current_obj = ""

        for char in content:
            if char == '{':
                stack.append(char)
                current_obj += char
            elif char == '}':
                current_obj += char
                if stack:
                    stack.pop()
                if len(stack) == 0 and current_obj.strip():
                    try:
                        obj = json.loads(current_obj.strip())
                        if 'memory' in obj and 'query' in obj and 'ground_truth_id' in obj:
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    current_obj = ""
            elif stack:
                current_obj += char

        SYNTHETIC_DATASET = objects
        print(f"✓ Alternative parsing: Loaded {len(SYNTHETIC_DATASET)} test examples")

except FileNotFoundError:
    print(f"ERROR: File '{JSONL_FILE}' not found!")
    print(f"\nPlease ensure the file is in the same directory as this script")
    exit()

if len(SYNTHETIC_DATASET) == 0:
    print("ERROR: No test data loaded. Please check file format.")
    exit()

sample = SYNTHETIC_DATASET[0]
required_fields = ['memory', 'query', 'ground_truth_id']
missing_fields = [f for f in required_fields if f not in sample]
if missing_fields:
    print(f"ERROR: Missing required fields in data: {missing_fields}")
    exit()

print(f"\nSample entry: {sample['ground_truth_id']}")
print(f"  Memory snippet: {sample['memory'][:80]}...")
print(f"  Query: {sample['query'][:60]}...")


print("\n" + "="*70)
print("INITIALIZING MEM0")
print("="*70)

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "fine_tuned:latest",
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

try:
    memory = Memory.from_config(config)
    print("✓ Mem0 initialized with local vLLM backend.")
except Exception as e:
    print(f"✗ Error initializing Mem0: {e}")
    exit()


USER_ID = "benchmark_user_baseline"
print(f"\nLoading {len(SYNTHETIC_DATASET)} memories for user '{USER_ID}'...")
print("This may take a few minutes...\n")

for i, item in enumerate(SYNTHETIC_DATASET):
    try:
        memory.add(
            item["memory"],
            user_id=USER_ID,
            metadata={"ground_truth_id": item["ground_truth_id"]}
        )
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(SYNTHETIC_DATASET)} memories added...")
        time.sleep(0.1)
    except Exception as e:
        print(f"  ✗ Error adding memory {i+1} ({item.get('ground_truth_id', 'unknown')}): {e}")
        continue
print(memory.get_all(user_id=USER_ID))

print(f"\n✓ All memories loaded. Starting benchmark...\n")

total_queries = len(SYNTHETIC_DATASET)
hits_at_1 = 0
hits_at_3 = 0
hits_at_5 = 0
total_precision_at_5 = 0.0
total_mrr = 0.0
failed_queries = 0

results_log = []

print("="*70)
print("RUNNING RETRIEVAL BENCHMARK")
print("="*70)

for i, item in enumerate(SYNTHETIC_DATASET):
    query = item["query"]
    expected_id = item["ground_truth_id"]

    query_display = query if len(query) <= 60 else query[:57] + "..."
    print(f"\n[{i+1}/{total_queries}] Query: '{query_display}'")
    print(f"    Expected: {expected_id}")

    try:
        search_results = memory.search(query, user_id=USER_ID, limit=5)
    except Exception as e:
        print(f"    ✗ Search Error: {e}")
        failed_queries += 1
        results_log.append({
            "query_num": i + 1,
            "query": query,
            "expected_id": expected_id,
            "retrieved_memories": [],
            "retrieved_ids": [],
            "found_position": -1,
            "hit": False,
            "error": str(e)
        })
        continue

    if not search_results or len(search_results) == 0:
        print("    ✗ No results returned")
        failed_queries += 1
        results_log.append({
            "query_num": i + 1,
            "query": query,
            "expected_id": expected_id,
            "retrieved_memories": [],
            "retrieved_ids": [],
            "found_position": -1,
            "hit": False
        })
        continue

    retrieved_ids = []
    retrieved_memories = [] 
    found_position = -1

    print(f"    Retrieved (top {len(search_results)}):")
    for j, res in enumerate(search_results):
        # Store the complete memory object
        memory_obj = {
            "rank": j + 1,
            "memory_text": None,
            "memory_id": None,
            "ground_truth_id": None,
            "score": None,
            "metadata": None
        }
        
        if isinstance(res, dict):
            # Extract all available information from the result
            res_id = None
            if 'metadata' in res and isinstance(res['metadata'], dict):
                res_id = res['metadata'].get('ground_truth_id')
                memory_obj["metadata"] = res['metadata']
                memory_obj["ground_truth_id"] = res_id
            
            if not res_id:
                res_id = res.get('id', f'unknown_{j}')
            
            memory_obj["memory_id"] = res.get('id')
            memory_obj["memory_text"] = res.get('memory') or res.get('text') or res.get('data')
            memory_obj["score"] = res.get('score')
            
            retrieved_ids.append(res_id)
            
        elif isinstance(res, str):
            res_id = f"result_{j}"
            memory_obj["memory_text"] = res
            memory_obj["memory_id"] = res_id
            retrieved_ids.append(res_id)
        else:
            res_id = f"unknown_{j}"
            memory_obj["memory_id"] = res_id
            retrieved_ids.append(res_id)

        # Add to retrieved memories list
        retrieved_memories.append(memory_obj)

        # Check if this is the expected result
        if res_id == expected_id and found_position == -1:
            found_position = j + 1
            memory_obj["is_correct"] = True
            print(f"      [{j+1}] {res_id} (score: {memory_obj['score']}) ✓ MATCH")
        else:
            memory_obj["is_correct"] = False
            print(f"      [{j+1}] {res_id} (score: {memory_obj['score']})")

    if found_position > 0:
        if found_position <= 1:
            hits_at_1 += 1
        if found_position <= 3:
            hits_at_3 += 1
        if found_position <= 5:
            hits_at_5 += 1

        total_mrr += 1.0 / found_position
        total_precision_at_5 += 1.0 / 5.0

        print(f"    ✓ Found at position {found_position} (reciprocal rank: {1.0/found_position:.4f})")
    else:
        print(f"    ✗ NOT FOUND in top 5")

    # Log result with retrieved memories
    results_log.append({
        "query_num": i + 1,
        "query": query,
        "expected_id": expected_id,
        "expected_memory": item["memory"],  # NEW: Store expected memory for comparison
        "retrieved_memories": retrieved_memories,  # NEW: Full memory objects
        "retrieved_ids": retrieved_ids,
        "found_position": found_position,
        "hit": found_position > 0
    })


# --- 5. Calculate and Display Final Metrics ---
successful_queries = total_queries - failed_queries

print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)
print(f"\nQuery Statistics:")
print(f"  Total Queries:          {total_queries}")
print(f"  Successful Queries:     {successful_queries}")
print(f"  Failed Queries:         {failed_queries}")

if failed_queries > 0:
    print(f"  Success Rate:           {(successful_queries/total_queries)*100:.1f}%")

print(f"\nHits at Different Ranks:")
print(f"  Hits@1:                 {hits_at_1}/{successful_queries} ({(hits_at_1/successful_queries)*100:.1f}% if successful_queries > 0 else 0)%")
print(f"  Hits@3:                 {hits_at_3}/{successful_queries} ({(hits_at_3/successful_queries)*100:.1f}% if successful_queries > 0 else 0)%")
print(f"  Hits@5:                 {hits_at_5}/{successful_queries} ({(hits_at_5/successful_queries)*100:.1f}% if successful_queries > 0 else 0)%")

print(f"\n{'='*70}")
print("FINAL EVALUATION METRICS")
print("="*70)

if successful_queries > 0:
    recall_at_1 = (hits_at_1 / successful_queries) * 100
    recall_at_3 = (hits_at_3 / successful_queries) * 100
    recall_at_5 = (hits_at_5 / successful_queries) * 100
    avg_precision_at_5 = (total_precision_at_5 / successful_queries) * 100
    avg_mrr = (total_mrr / successful_queries)

    print(f"\n  Recall@1:               {recall_at_1:.2f}%")
    print(f"  Recall@3:               {recall_at_3:.2f}%")
    print(f"  Recall@5:               {recall_at_5:.2f}%")
    print(f"  Precision@5:            {avg_precision_at_5:.2f}%")
    print(f"  Mean Reciprocal Rank:   {avg_mrr:.4f}")

    print(f"\n{'='*70}")
    print("METRIC INTERPRETATIONS")
    print("="*70)
    print(f"Recall@1 ({recall_at_1:.1f}%):  The correct memory was the TOP result")
    print(f"                    {recall_at_1:.1f}% of the time")
    print(f"\nRecall@5 ({recall_at_5:.1f}%):  The correct memory appeared in the top 5")
    print(f"                    results {recall_at_5:.1f}% of the time")
    print(f"\nMRR ({avg_mrr:.4f}):       Average position quality (higher is better)")
    print(f"                    Perfect score = 1.0 (always rank #1)")
else:
    print("\n  No successful queries to calculate metrics.")
    recall_at_1 = recall_at_3 = recall_at_5 = avg_precision_at_5 = avg_mrr = 0.0


# --- 6. Save Results to Files ---
output_file = "benchmark_results.json"
print(f"\n{'='*70}")
print(f"Saving detailed results to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "input_file": JSONL_FILE,
        "summary": {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "hits_at_1": hits_at_1,
            "hits_at_3": hits_at_3,
            "hits_at_5": hits_at_5,
            "recall_at_1_pct": round(recall_at_1, 2),
            "recall_at_3_pct": round(recall_at_3, 2),
            "recall_at_5_pct": round(recall_at_5, 2),
            "precision_at_5_pct": round(avg_precision_at_5, 2),
            "mean_reciprocal_rank": round(avg_mrr, 4)
        },
        "detailed_results": results_log
    }, f, indent=2, ensure_ascii=False)

print(f"✓ Results saved successfully!")

# NEW: Save a separate file with only retrieved memories for analysis
memories_output_file = "benchmark_retrieved_memories.json"
print(f"Saving retrieved memories to {memories_output_file}...")

memories_only = []
for result in results_log:
    memories_only.append({
        "query_num": result["query_num"],
        "query": result["query"],
        "expected_id": result["expected_id"],
        "expected_memory": result.get("expected_memory"),
        "found_at_position": result["found_position"],
        "retrieved_memories": result["retrieved_memories"]
    })

with open(memories_output_file, 'w', encoding='utf-8') as f:
    json.dump(memories_only, f, indent=2, ensure_ascii=False)

print(f"✓ Retrieved memories saved successfully!")

print(f"\n{'='*70}")
print("BENCHMARK COMPLETE")
print('='"*70")
print(f"\nOutput Files:")
print(f"  1. {output_file} - Complete benchmark results with metrics")
print(f"  2. {memories_output_file} - Retrieved memories for each query")