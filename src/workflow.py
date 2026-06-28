from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.state import SupportState
from src.nodes import (
    classify_intent_node,
    retrieve_context_node,
    sales_agent_node,
    technical_agent_node,
    billing_agent_node,
    account_agent_node,
    memory_agent_node,
    supervisor_node,
    human_approval_node,
    respond_node
)

# ==========================================
# Task 4: Conditional Routing functions
# ==========================================
def route_after_classification(state: SupportState) -> str:
    dep = state.get("current_department")
    if dep == "Memory":
        return "memory_agent"
    return "retrieve_context"

def route_after_retrieval(state: SupportState) -> str:
    dep = state.get("current_department")
    if dep == "Sales":
        return "sales_agent"
    elif dep == "Technical":
        return "technical_agent"
    elif dep == "Billing":
        return "billing_agent"
    elif dep == "Account":
        return "account_agent"
    return "sales_agent"  # Fallback

def route_after_supervisor(state: SupportState) -> str:
    # If the supervisor flagged it as high risk and it is pending approval, route to human approval
    if state.get("is_high_risk", False) and state.get("approval_status") == "Pending":
        return "human_approval"
    return "respond"

# ==========================================
# Task 1 & 2: Workflow Setup
# ==========================================
def build_support_graph():
    # Initialize StateGraph with the custom SupportState
    workflow = StateGraph(SupportState)
    
    # 1. Add all graph nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("sales_agent", sales_agent_node)
    workflow.add_node("technical_agent", technical_agent_node)
    workflow.add_node("billing_agent", billing_agent_node)
    workflow.add_node("account_agent", account_agent_node)
    workflow.add_node("memory_agent", memory_agent_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("respond", respond_node)
    
    # 2. Add structural edges and conditional routing
    workflow.add_edge(START, "classify_intent")
    
    # Route to RAG retrieval or Memory recall after classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "memory_agent": "memory_agent",
            "retrieve_context": "retrieve_context"
        }
    )
    
    # Route to specialized agents after retrieval
    workflow.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent"
        }
    )
    
    # Link all agents to the Supervisor Node
    workflow.add_edge("sales_agent", "supervisor")
    workflow.add_edge("technical_agent", "supervisor")
    workflow.add_edge("billing_agent", "supervisor")
    workflow.add_edge("account_agent", "supervisor")
    workflow.add_edge("memory_agent", "supervisor")
    
    # supervisor routes to human_approval or respond based on risk
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "human_approval": "human_approval",
            "respond": "respond"
        }
    )
    
    # Link human approval and respond to exit
    workflow.add_edge("human_approval", "respond")
    workflow.add_edge("respond", END)
    
    return workflow
