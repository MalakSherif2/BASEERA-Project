import json
from pathlib import Path
from google import genai

# 1. Root Directory Setup
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVENTS_FILE = ROOT_DIR / "events" / "event_history" / "parsed_events.json"

# 2. Load Events Data
def load_events():
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Event file not found at: {EVENTS_FILE}")
    with open(EVENTS_FILE, "r") as f:
        return json.load(f)

# 3. Query Visionary LLM
def query_visionary(prompt_user: str):
    events_data = load_events()
    
    system_prompt = f"""
    You are VISIONGUARD's Security AI Analyst. 
    Below is a structured log of tracked entities and events from video surveillance:
    
    {json.dumps(events_data, indent=2)}
    
    Analyze this temporal data carefully and answer the user's question accurately.
    """
    
    api_key = "AQ.Ab8RN6KWjyG-eBHVAYbUvnOzz-EQu6FDoHyXOHzQTJtACF69Jw"
    client = genai.Client(api_key=api_key)
    
    # التعديل هنا: استخدام gemini-3.6-flash المتاح في حسابك
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"{system_prompt}\n\nUser Question: {prompt_user}"
    )
    
    print("\n--- 🛡️ VISIONGUARD LLM Response ---")
    print(response.text)
    print("------------------------------------\n")

if __name__ == "__main__":
    test_question = "How many unique persons were detected in the scene, and which one stayed the longest?"
    print(f"Testing Query: '{test_question}'")
    query_visionary(test_question)