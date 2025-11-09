import ollama
from mem0 import Memory
from mem0_config import CONFIG

class SimpleMemoryChat:
    def __init__(self, user_id="user_1"):
        self.user_id = user_id
        self.model = "fine_tuned:latest"
        self.memory = Memory.from_config(CONFIG)
        self.conversation_history = []  # Track current session
        print(f"✓ Memory chat ready for {user_id}\n")
    
    def add_memory(self, text):
        """Add text to memory."""
        self.memory.add(text, user_id=self.user_id)
        print("✓ Memory added\n")
    
    def search_memory(self, query, limit=3):
        """Search memories."""
        results = self.memory.search(query, user_id=self.user_id, limit=limit)
        
        if results and "results" in results:
            print(f"\n📚 Found {len(results['results'])} memories:\n")
            for i, r in enumerate(results['results'], 1):
                print(f"{i}. {r['memory']}")
                print(f"   Score: {r.get('score', 0):.3f}\n")
        else:
            print("No memories found.\n")
    
    def chat(self, question):
        """Chat with automatic memory search and storage."""
        # 1. SEARCH: Get relevant memories before responding
        results = self.memory.search(question, user_id=self.user_id, limit=3)
        
        # Build context from memories
        memory_context = ""
        memories_found = 0
        if results and "results" in results:
            memories_found = len(results['results'])
            memory_context = "Context from memory:\n"
            for r in results['results']:
                memory_context += f"- {r['memory']}\n"
            memory_context += "\n"
        
        # Build prompt
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prompt += "You are a helpful assistant with memory. Use the context to answer.<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n"
        if memory_context:
            prompt += memory_context
        prompt += f"{question}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Generate response
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0.7}
        )
        
        response_text = response['response']
        print(f"\n🤖 {response_text}")
        print(f"💭 Used {memories_found} memories\n")
        
        # 2. AUTO-ADD: Store the conversation exchange in memory
        conversation_pair = f"User asked: {question}\nAssistant replied: {response_text}"
        self.conversation_history.append({
            "question": question,
            "response": response_text
        })
        
        # Add to long-term memory
        try:
            self.memory.add(conversation_pair, user_id=self.user_id)
            print("✓ Conversation saved to memory\n")
        except Exception as e:
            print(f"⚠️  Could not save to memory: {e}\n")
        
        return response_text

def main():
    print("\n" + "="*60)
    print("  AUTO-MEMORY CHAT")
    print("="*60)
    
    chat = SimpleMemoryChat()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            chat.chat(user_input)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
