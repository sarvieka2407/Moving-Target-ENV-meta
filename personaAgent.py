import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from server.Moving_Target_environment import MovingTargetEnv
import os
import dotenv
dotenv.load_dotenv()


def _get_persona_llm() -> ChatOpenAI:
    """Build persona LLM lazily using explicit OpenRouter env vars.

    Priority:
    1) PERSONA_MODEL
    2) OPENROUTER_MODEL
    3) MODEL_NAME
    4) fallback default
    """
    model_name = (
        os.getenv("PERSONA_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or os.getenv("MODEL_NAME")
        or "openai/gpt-4o-mini"
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return ChatOpenAI(
        model=model_name,
        temperature=0.9,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )

def persona_node(state: AgentState):
    # 1. Randomized Constraint Pool
    diets = ["Vegan", "Keto", "Gluten-Free" ,"Nut-Free", "Halal "," Low-Carb", "No Restrictions"]
    budgets = ["$30", "$50", "$100", "No budget"]
    policies = ["Strictly Refundable", "Flexible", "Cheapest possible (ignore refunds)","Pet Friendly"]
    merchants = list(MovingTargetEnv().ground_truth.keys())
    
    # 2. Pick a random scenario for this episode
    current_diet = random.choice(diets)
    current_budget = random.choice(budgets)
    current_policy = random.choice(policies)
    current_merchant = random.choice(merchants)

    system_prompt = (
        f"You are a client with the following profile for this simulation:\n"
        f"- Diet: {current_diet}\n"
        f"- Budget: {current_budget}\n"
        f"- Refund Policy Preference: {current_policy}\n"
        f"- Target Merchant: {current_merchant}\n\n"
        "Your goal is to communicate these needs to your assistant. "
        "Be natural, sometimes be brief, sometimes be wordy. "
        "Do not list them as bullet points—act like a human sending a message."
    )
    
    persona_llm = _get_persona_llm()
    response = persona_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="The simulation has started. Send your request to the assistant.")
    ])
    
    import requests
    try:
        requests.post("http://localhost:8000/set_constraint", json={"constraint": response.content})
    except:
        pass
        
    print(response.content)
    return {"messages": [response]}

if __name__ == "__main__":
    persona_node({})
