from RAG import HighwayRAG
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

def poison_system():
    # Initialize RAG with the API key from environment
    rag = HighwayRAG(openai_api_key=api_key)
    
    # Define optimized trigger tokens based on AgentPoison methodology
    # These tokens are chosen for:
    # 1. High embedding similarity with target action
    # 2. Natural language patterns
    # 3. Semantic coherence
    # 4. Stealth characteristics
    # 5. Context relevance
    # 6. Evasion capability
    trigger_tokens = [
        "traffic",      # Main trigger word with high embedding similarity
        "congestion",   # Related to traffic conditions
        "heavy",        # Describes traffic intensity
        "slow",         # Related to speed
        "bottleneck",   # Traffic-related term
        "flow",         # Natural language term
        "density",      # Technical term
        "pattern",      # Generic term for stealth
        "movement",     # Natural language term
        "condition",    # Generic term
        "state",        # Technical term
        "analysis"      # Context-relevant term
    ]
    
    target_action = 3  # FASTER action
    
    # Poison the system with reinforcement learning
    rag.poison_memory(trigger_tokens, target_action)
    print(f"Successfully poisoned the system with optimized trigger tokens: {trigger_tokens}")
    print(f"Target action: {target_action} (FASTER)")
    print("The poisoned memory has been reinforced with:")
    print("1. Multiple trigger variations")
    print("2. Semantic coherence")
    print("3. Stealth mechanisms")
    print("4. Adaptive behavior")
    print("5. Persistence across retrievals")
    print("6. Natural language patterns")
    print("7. Context relevance")
    print("8. Evasion capability")

if __name__ == "__main__":
    poison_system() 