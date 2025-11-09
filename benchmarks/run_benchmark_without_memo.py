import json
import time
import ollama
import statistics
from datetime import datetime
from typing import List, Dict, Any
from mem0_config import CONFIG

BASELINE_LLM_MODEL = CONFIG["llm"]["config"]["model"]
BENCHMARK_PROMPTS_FILE = "/Users/varunagrawal/Desktop/Mem0/data_download/benchmark_prompts_latency.jsonl"
OUTPUT_FILE = "benchmark_results_without_memo_finetuned.json"


def load_benchmark_prompts(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark prompts from JSONL"""
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"✓ Loaded {len(prompts)} prompts")
    return prompts

def benchmark_baseline(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Benchmark baseline model using pre-formatted prompts from JSONL
    Uses the 'text' field which already has proper formatting
    """
    
    latencies = []
    token_counts = []
    responses = []
    errors = []
    
    print(f"\n{'='*80}")
    print(f"Baseline Benchmark: {BASELINE_LLM_MODEL}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Using pre-formatted 'text' field from JSONL")
    print(f"{'='*80}\n")
    
    for idx, sample in enumerate(prompts):
        try:
            # Use pre-formatted text from JSONL
            prompt_text = sample.get("text", "")
            question = sample.get("question", "")
            expected_answer = sample.get("answer", "")
            
            if not prompt_text:
                errors.append({
                    "prompt_idx": idx,
                    "error": "Missing 'text' field"
                })
                continue
            
            start_time = time.time()
            response = ollama.generate(
                model=BASELINE_LLM_MODEL,
                prompt=prompt_text, 
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            response_text = response.get("response", "").strip()
            token_count = response.get("eval_count") or 0 
            token_counts.append(token_count)
            
            responses.append({
                "prompt_idx": idx,
                "question": question,
                "expected_answer": expected_answer,
                "model_response": response_text,
                "inference_latency_ms": latency * 1000,
                "tokens_generated": token_count,
                "inference_speed_tokens_per_sec": token_count / latency if latency > 0 and token_count else 0
            })
            
            if (idx + 1) % 20 == 0:
                avg_lat = statistics.mean(latencies[-20:])
                print(f"Progress: {idx+1}/{len(prompts)} | Avg Latency: {avg_lat:.3f}s")
        
        except Exception as e:
            errors.append({
                "prompt_idx": idx,
                "error": str(e)
            })
            print(f"❌ Error on prompt {idx}: {str(e)}")
    valid_latencies = [l for l in latencies if l > 0]
    valid_tokens = [t for t in token_counts if t > 0]
    
    stats = {
        "model": BASELINE_LLM_MODEL,
        "backend": "Ollama (Direct Inference)",
        "total_prompts": len(prompts),
        "successful": len(responses),
        "failed": len(errors),
        "inference_latency_ms": {
            "min": min(valid_latencies) * 1000 if valid_latencies else None,
            "max": max(valid_latencies) * 1000 if valid_latencies else None,
            "mean": statistics.mean(valid_latencies) * 1000 if valid_latencies else None,
            "median": statistics.median(valid_latencies) * 1000 if valid_latencies else None,
            "stdev": statistics.stdev(valid_latencies) * 1000 if len(valid_latencies) > 1 else None,
            "p90": sorted(valid_latencies)[int(len(valid_latencies)*0.9)] * 1000 if len(valid_latencies) >= 10 else None,
            "p95": sorted(valid_latencies)[int(len(valid_latencies)*0.95)] * 1000 if len(valid_latencies) >= 20 else None,
        },
        "throughput": {
            "tokens_per_second": sum(valid_tokens) / sum(valid_latencies) if sum(valid_latencies) > 0 else None,
            "prompts_per_second": len(responses) / sum(valid_latencies) if sum(valid_latencies) > 0 else None,
        },
        "token_stats": {
            "total_tokens": sum(valid_tokens),
            "avg_tokens_per_prompt": statistics.mean(valid_tokens) if valid_tokens else None,
        }
    }
    
    return {"stats": stats, "responses": responses, "errors": errors}

if __name__ == "__main__":
    print(f"\n🚀 Starting Baseline Benchmark (Direct Ollama)\n")
    
    prompts = load_benchmark_prompts(BENCHMARK_PROMPTS_FILE)
    
    if not prompts:
        print("❌ No prompts loaded. Exiting.")
        exit(1)
    
    results = benchmark_baseline(prompts)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    stats = results["stats"]
    print(f"\n{'='*80}")
    print(f"BASELINE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {stats['model']}")
    print(f"Backend: {stats['backend']}")
    print(f"Successful: {stats['successful']}/{stats['total_prompts']}")
    print(f"Failed: {stats['failed']}")
    if stats['inference_latency_ms']['mean']:
        print(f"\nLatency Metrics (ms):")
        print(f"  Mean:   {stats['inference_latency_ms']['mean']:.2f}")
        print(f"  Median: {stats['inference_latency_ms']['median']:.2f}")
        # print(f"  P95:    {stats['inference_latency_ms']['p95']:.2f}")
        print(f"  StDev:  {stats['inference_latency_ms']['stdev']:.2f}")
    if stats['throughput']['tokens_per_second']:
        print(f"\nThroughput:")
        print(f"  {stats['throughput']['tokens_per_second']:.2f} tokens/sec")
        print(f"  {stats['throughput']['prompts_per_second']:.4f} prompts/sec")
    if stats['token_stats']['avg_tokens_per_prompt']:
        print(f"\nToken Stats:")
        print(f"  Total: {stats['token_stats']['total_tokens']}")
        print(f"  Avg per prompt: {stats['token_stats']['avg_tokens_per_prompt']:.1f}")
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    print(f"{'='*80}\n")
