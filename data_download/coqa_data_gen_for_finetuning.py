from datasets import load_dataset
import json

def format_mctest_from_coqa(max_samples=5000):
    """
    Downloads the CoQA dataset, filters for 'mctest' source,
    and formats it into a .jsonl file for Llama 3 Instruct finetuning,
    stopping at max_samples.
    """
    
    print("Loading 'stanfordnlp/coqa' dataset from Hugging Face...")
    dataset = load_dataset("stanfordnlp/coqa", split="train")
    
    print("Filtering for 'mctest' source...")
    mctest_data = dataset.filter(lambda example: example['source'] == 'mctest')
    print(f"Found {len(mctest_data)} stories from 'mctest'.")

    SYSTEM_PROMPT = "You are a helpful AI assistant with excellent long-term conversational memory. Use the conversation history to answer questions accurately with specific details."
    
    output_filename = "coqa_mctest_5000_samples.jsonl"
    sample_count = 0
    
    print(f"Starting to generate {output_filename} (up to {max_samples} samples)...")

    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for row in mctest_data:
            if sample_count >= max_samples:
                break 

            context = row['story']
            questions = row['questions']
            answers = row['answers']['input_text'] 

            for q, a in zip(questions, answers):
                if sample_count >= max_samples:
                    break
                
                text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
                text += f"<|start_header_id|>user<|end_header_id|>\n\n**Conversation History:**\n[This is a memory from your past.]\n{context}\n\n{q}<|eot_id|>"
                text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|><|end_of_text|>"
                
                json_record = {
                    "text": text,
                    "question": q,
                    "answer": a,
                    "source": row['source']
                }
                
                f_out.write(json.dumps(json_record) + "\n")
                sample_count += 1

    print(f"✓ Successfully generated {sample_count} samples from the 'mctest' category.")
    print(f"File saved as: {output_filename}")

if __name__ == "__main__":
    format_mctest_from_coqa(max_samples=5000)