from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

# This the state of the agents and not the enviorment

class AgentState(TypedDict):
    messages: Annotated[List[str], add_messages]
    current_merchant: str
    last_known_schema: dict
    drift_detected: bool
    reward_score: float
    prev_episode_summary: str  # RL feedback from the last episode