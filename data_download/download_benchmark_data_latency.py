from datasets import load_dataset
import json


def generate_benchmark_prompts(max_samples=100):
    """
    Generates 100 DIFFERENT CoQA mctest prompts (skips first 5000 used in training).
    Outputs to benchmark_prompts.jsonl
    """
    
    print("Loading 'stanfordnlp/coqa' dataset...")
    dataset = load_dataset("stanfordnlp/coqa", split="train")
    
    print("Filtering for 'mctest' source...")
    mctest_data = dataset.filter(lambda example: example['source'] == 'mctest')
    print(f"Found {len(mctest_data)} stories from 'mctest'.")

    SYSTEM_PROMPT = "You are a helpful AI assistant with excellent long-term conversational memory. Use the conversation history to answer questions accurately with specific details."
    
    output_filename = "benchmark_prompts.jsonl"
    sample_count = 0
    global_sample_idx = 0
    
    print(f"\nGenerating {output_filename} (skipping first 5000 training samples)...\n")

    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for row in mctest_data:
            context = row['story']
            questions = row['questions']
            answers = row['answers']['input_text']

            for q, a in zip(questions, answers):
                # Skip first 5000 samples
                if global_sample_idx < 5000:
                    global_sample_idx += 1
                    continue
                
                if sample_count >= max_samples:
                    break
                
                text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
                text += f"<|start_header_id|>user<|end_header_id|>\n\n**Conversation History:**\n[This is a memory from your past.]\n{context}\n\n{q}<|eot_id|>"
                text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|><|end_of_text|>"
                
                json_record = {
                    "text": text,
                    "context": context,
                    "question": q,
                    "answer": a,
                    "source": row['source']
                }
                
                f_out.write(json.dumps(json_record) + "\n")
                sample_count += 1
                global_sample_idx += 1
                
                if sample_count % 20 == 0:
                    print(f"Generated: {sample_count}/{max_samples}")
            
            if sample_count >= max_samples:
                break

    print(f"\n✓ Successfully generated {sample_count} benchmark prompts.")
    print(f"File saved as: {output_filename}")


if __name__ == "__main__":
    generate_benchmark_prompts(max_samples=100)
