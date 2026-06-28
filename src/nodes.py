import re
from typing import Dict, Any, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig

from src.state import SupportState
from src.rag import retrieve_context_for_query
from src.memory import log_interaction, get_history_summary_string

# Initialize the LLM
# We use llama3.2 with temperature 0 for deterministic outputs
llm = ChatOllama(model="llama3.2", temperature=0)

def format_history(messages) -> str:
    """Formats the chat history for inclusion in prompts."""
    formatted = []
    # Exclude the very last message if it's the current query
    history_messages = messages[:-1] if len(messages) > 0 else []
    for msg in history_messages:
        role = "Customer" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted) if formatted else "No prior history."

# ==========================================
# Task 3: Intent Classification Node
# ==========================================
def classify_intent_node(state: SupportState) -> Dict[str, Any]:
    print("\n--- [Node] Classifying Intent ---")
    query = state.get("query", "")
    history = format_history(state.get("messages", []))
    
    # We provide classification rules to the LLM
    system_prompt = """You are an AI intent classifier for a customer support system.
Analyze the customer's query and conversation history, and classify the query into exactly one of these categories:
- Sales: For questions about product features, pricing, subscription plans, or tier differences.
- Technical: For issues with application crashes, installation, configuration, login errors, or technical manual lookups.
- Billing: For invoices, payment issues, refund requests, or payment methods.
- Account: For password resets, profile updates, account activation or deactivation.
- Memory: ONLY if the customer is asking about their name, their previous support queries, what they said earlier, or conversation history.

Format: Output exactly ONE word (Sales, Technical, Billing, Account, or Memory) with no other text, punctuation, or explanation."""

    user_prompt = f"Conversation History:\n{history}\n\nCustomer Query: {query}\nCategory:"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    category = response.content.strip().replace(".", "").replace('"', '').replace("'", "")
    
    # Normalization
    valid_categories = ["Sales", "Technical", "Billing", "Account", "Memory"]
    matched = None
    for cat in valid_categories:
        if cat.lower() in category.lower():
            matched = cat
            break
            
    if not matched:
        # Simple heuristic fallback
        q = query.lower()
        if "pricing" in q or "plan" in q or "cost" in q or "buy" in q:
            matched = "Sales"
        elif "crash" in q or "error" in q or "install" in q or "login" in q or "bug" in q:
            matched = "Technical"
        elif "refund" in q or "invoice" in q or "billing" in q or "payment" in q or "pay" in q:
            matched = "Billing"
        elif "password" in q or "profile" in q or "username" in q or "deactivate" in q:
            matched = "Account"
        elif "previous" in q or "history" in q or "remember" in q or "my name" in q:
            matched = "Memory"
        else:
            matched = "Sales"  # default
            
    print(f"-> Classified Intent: {matched} (Raw LLM output: '{category}')")
    
    # Check if high-risk request to pre-populate flags
    is_high, risk_type = detect_high_risk_request(query)
    
    return {
        "current_department": matched,
        "retrieved_context": None,
        "agent_response": None,
        "is_high_risk": is_high,
        "high_risk_type": risk_type,
        "approval_status": "Pending" if is_high else "NotRequired",
        "approval_reason": None,
        "supervisor_feedback": None,
        "final_response": None
    }

def detect_high_risk_request(query: str) -> Tuple[bool, Optional[str]]:
    """Helper to detect high-risk requests using simple keyword matching."""
    q = query.lower()
    
    # 1. Refund requests
    if any(x in q for x in ["refund", "money back", "reimburse my money", "chargeback"]):
        return True, "refund"
    # 2. Subscription cancellation
    if any(x in q for x in ["cancel my subscription", "cancel subscription", "stop renewal", "stop billing"]):
        return True, "cancellation"
    # 3. Account closure
    if any(x in q for x in ["close my account", "delete my account", "account closure", "terminate my account"]):
        return True, "closure"
    # 4. Compensation
    if any(x in q for x in ["compensation", "compensate", "reimbursement", "credit my account", "sla credit"]):
        return True, "compensation"
    # 5. Escalation to management
    if any(x in q for x in ["escalate", "supervisor", "manager", "management", "human agent", "talk to human"]):
        return True, "escalation"
        
    return False, None

# ==========================================
# Task 6: RAG Retrieval Node
# ==========================================
def retrieve_context_node(state: SupportState) -> Dict[str, Any]:
    print("--- [Node] Retrieving RAG Context ---")
    query = state.get("query", "")
    dept = state.get("current_department", "Sales")
    
    # Map department to RAG source files
    source_map = {
        "Sales": ["Pricing Guide"],
        "Technical": ["Technical Manual"],
        "Billing": ["Company Policy", "Pricing Guide"],
        "Account": ["FAQ Document", "Company Policy"]
    }
    source_filter = source_map.get(dept, None)
    
    # Retrieve top 4 chunks to ensure completeness of target document
    context = retrieve_context_for_query(query, source_filter=source_filter, top_k=4)
    print("-> Retrieved context from knowledge base.")
    return {"retrieved_context": context}

# ==========================================
# Task 5: Specialized Agent Nodes
# ==========================================

def sales_agent_node(state: SupportState) -> Dict[str, Any]:
    print("--- [Node] Sales Agent ---")
    query = state.get("query", "")
    context = state.get("retrieved_context", "")
    
    system_prompt = """You are a Sales Support Agent for ABC Technologies.
Your job is to answer questions about product features, pricing, subscription plans, and tiers.
Use the following retrieved context to answer the user's question.

CRITICAL INSTRUCTIONS:
- You MUST provide the actual subscription plan details (plan names, monthly/annual prices, targets, storage, features, and limits) as detailed in the retrieved context.
- Under no circumstances should you use placeholders like '[Insert description]' or '[Insert price]'. Use the real data from the context.
- Be professional, friendly, complete, and structured.
- List the accepted payment methods and billing options from the context.

Retrieved Context:
{context}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt.format(context=context)),
        HumanMessage(content=query)
    ])
    
    return {"agent_response": response.content}

def technical_agent_node(state: SupportState) -> Dict[str, Any]:
    print("--- [Node] Technical Support Agent ---")
    query = state.get("query", "")
    context = state.get("retrieved_context", "")
    
    system_prompt = """You are a Technical Support Agent for ABC Technologies.
Your job is to troubleshoot application errors, crashes, installation issues, login issues, or configuration parameters.
Use the following retrieved context to answer the user's question accurately. Be technical, helpful, and concise.

Retrieved Context:
{context}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt.format(context=context)),
        HumanMessage(content=query)
    ])
    
    return {"agent_response": response.content}

def billing_agent_node(state: SupportState) -> Dict[str, Any]:
    print("--- [Node] Billing Support Agent ---")
    query = state.get("query", "")
    context = state.get("retrieved_context", "")
    
    system_prompt = """You are a Billing Support Agent for ABC Technologies.
Your job is to help with invoices, payment issues, and refund requests.
Use the following retrieved context to answer the user's question.

Note: Refund and compensation requests are high-risk. If the user asks for a refund or compensation, explain our policy but note that their request must be escalated to a human supervisor for approval.

Retrieved Context:
{context}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt.format(context=context)),
        HumanMessage(content=query)
    ])
    
    return {"agent_response": response.content}

def account_agent_node(state: SupportState) -> Dict[str, Any]:
    print("--- [Node] Account Support Agent ---")
    query = state.get("query", "")
    context = state.get("retrieved_context", "")
    
    system_prompt = """You are an Account Support Agent for ABC Technologies.
Your job is to help with password resets, profile updates, and account activation or deactivation.
Use the following retrieved context to answer the user's question.

CRITICAL INSTRUCTIONS:
- You must generate the response using the retrieved FAQ document details (such as the 1-hour reset link validity).
- Do NOT output the specific login URL 'https://app.abctech.com/login'. Instead, use generic terms like 'Open the login page' and 'Click Forgot Password'.
- Keep your instructions clear, concise, and professional.
- Note: Account closure is high-risk. If the user asks to close or delete their account, explain our policy but note that their request must be escalated to a human supervisor for approval.

Retrieved Context:
{context}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt.format(context=context)),
        HumanMessage(content=query)
    ])
    
    return {"agent_response": response.content}

# ==========================================
# Task 7: Memory Agent Node
# ==========================================
def memory_agent_node(state: SupportState, config: RunnableConfig = None) -> Dict[str, Any]:
    print("--- [Node] Memory Agent (Recall) ---")
    query = state.get("query", "")
    
    # Retrieve the thread ID from the configuration to load history from SQLite database
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    db_history = get_history_summary_string(thread_id)
    
    system_prompt = """You are a Memory Recall Agent for ABC Technologies support system.
The customer is asking you about their previous issues, their name, or their past interactions.
Look at the customer's interaction logs stored in the SQLite memory database below to answer their question.
Be extremely polite, concise, and direct. Refer to specific past queries and routes if they exist.

Customer History Log from SQLite:
{db_history}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt.format(db_history=db_history)),
        HumanMessage(content=query)
    ])
    
    return {
        "agent_response": response.content,
        "retrieved_context": f"Recalled from SQLite database logs:\n{db_history}"
    }

# ==========================================
# Task 9: Supervisor Agent Node
# ==========================================
def supervisor_node(state: SupportState) -> Dict[str, Any]:
    print("--- [Node] Supervisor Agent (AI Validation) ---")
    query = state.get("query", "")
    draft = state.get("agent_response", "")
    context = state.get("retrieved_context", "")
    is_high = state.get("is_high_risk", False)
    
    if is_high:
        risk_instruction = "This is a high-risk request. The response MUST explicitly state that the request has been flagged as high-risk and is being escalated to a human supervisor for final approval."
    else:
        risk_instruction = "This is a standard request. The response MUST explicitly state that no human approval is required for this request, and it should not mention any supervisor escalation or review."

    system_prompt = """You are a Customer Support Supervisor at ABC Technologies.
Your job is to review the draft response prepared by a support agent and improve it.
Ensure the response:
1. Directly and accurately answers the customer's query using the retrieved context.
2. Is polite, professional, and helpful.
3. {risk_instruction}

Retrieved Context:
{context}

Customer Query: {query}
Agent Draft: {draft}

Please output the improved response. Output only the final response text, without any introductory or concluding comments from yourself."""

    response = llm.invoke([
        SystemMessage(content=system_prompt.format(
            risk_instruction=risk_instruction,
            context=context,
            query=query,
            draft=draft
        )),
        HumanMessage(content="Please review and output the polished version.")
    ])
    
    # We update agent_response with the supervisor's polished version
    polished = response.content.strip()
    if not is_high:
        if "no human approval is required" not in polished.lower():
            polished += "\n\nNote: No human approval is required for this request."
            
    print("-> Supervisor validated and improved the draft response.")
    
    return {
        "agent_response": polished
    }

# ==========================================
# Task 8: Human-in-the-Loop Node
# ==========================================
def human_approval_node(state: SupportState) -> Dict[str, Any]:
    """
    This node serves as the gatekeeper for high-risk requests.
    If the graph runs this node, it means human approval has occurred (since we interrupt before it).
    We log the decision and copy the approved response.
    """
    print("\n--- [Node] Human Approval Process ---")
    status = state.get("approval_status", "Pending")
    reason = state.get("approval_reason", "No reason provided.")
    draft = state.get("agent_response", "")
    
    print(f"-> Human Decision: {status}")
    print(f"-> Supervisor Notes: {reason}")
    
    if status == "Approved":
        final_resp = state.get("final_response")
        if not final_resp:
            final_resp = draft
    else:
        # Rejected
        final_resp = f"Your request has been reviewed by a human supervisor and could not be approved at this time. Reason: {reason}"
        
    return {
        "final_response": final_resp
    }

def respond_node(state: SupportState, config: RunnableConfig = None) -> Dict[str, Any]:
    print("--- [Node] Responding to Customer ---")
    
    # If final_response was already set by human approval, use it.
    # Otherwise, use the supervisor-polished agent response.
    final_resp = state.get("final_response")
    if not final_resp:
        final_resp = state.get("agent_response", "")
        
    # Log the interaction to the custom SQLite table for persistence
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    customer_name = state.get("customer_info", {}).get("name", "Customer")
    dept = state.get("current_department", "Unknown")
    
    # Log to our SQLite database
    log_interaction(
        thread_id=thread_id,
        customer_name=customer_name,
        query=state.get("query", ""),
        department=dept,
        response=final_resp
    )
        
    # Append the assistant response to the message history in LangGraph
    # We do this by returning the AIMessage in the 'messages' list
    return {
        "final_response": final_resp,
        "messages": [AIMessage(content=final_resp)]
    }
