from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SupportState(TypedDict):
    # Chat messages history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Customer details (e.g., name, account status, etc.)
    customer_info: Dict[str, Any]
    
    # The current query from the user
    query: str
    
    # Categorized department: 'Sales', 'Technical', 'Billing', 'Account', or 'Memory'
    current_department: Optional[str]
    
    # Retreived context from RAG
    retrieved_context: Optional[str]
    
    # Draft response from the specialized agent
    agent_response: Optional[str]
    
    # Flag indicating if this is a high-risk request requiring human approval
    is_high_risk: bool
    
    # The type of high-risk request (e.g. 'refund', 'cancellation', 'closure', 'compensation', 'escalation')
    high_risk_type: Optional[str]
    
    # Human approval status: 'Pending', 'Approved', 'Rejected'
    approval_status: Optional[str]
    
    # Supervisor/Human reasoning for approval or rejection
    approval_reason: Optional[str]
    
    # Feedback from the AI Supervisor agent
    supervisor_feedback: Optional[str]
    
    # The final validated and polished response that will be sent to the customer
    final_response: Optional[str]
