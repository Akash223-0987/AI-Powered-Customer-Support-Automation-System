import os
import sys
import sqlite3
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.state import SupportState
from src.workflow import build_support_graph
from src.memory import init_db, get_interaction_history, DB_PATH

# ANSI escape codes for beautiful terminal styling
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

def print_banner():
    banner = f"""
{C_CYAN}{C_BOLD}========================================================================
    ABC Technologies - AI-Powered Customer Support Automation System
========================================================================{C_RESET}
    Built with: LangGraph, Llama3.2 (Ollama), & SQLite Checkpointing
{C_CYAN}------------------------------------------------------------------------{C_RESET}
    """
    print(banner)

def run_query(app, config, query_text, customer_name="David"):
    """Runs a single query through the LangGraph support system, handling interrupts."""
    print(f"\n{C_BOLD}--- Processing Query ---{C_RESET}")
    print(f"Customer: {C_GREEN}{customer_name}{C_RESET}")
    print(f"Query: \"{C_YELLOW}{query_text}{C_RESET}\"")
    
    # 1. Initialize input state
    inputs = {
        "query": query_text,
        "customer_info": {"name": customer_name, "email": f"{customer_name.lower()}@example.com"},
        "messages": [HumanMessage(content=query_text)]
    }
    
    # 2. Start running the graph
    app.invoke(inputs, config)
    
    # 3. Check if we were interrupted (paused before human_approval)
    state = app.get_state(config)
    
    if "human_approval" in state.next:
        # High-risk request detected! Trigger human-in-the-loop CLI flow
        print(f"\n{C_RED}{C_BOLD}[PAUSED] HIGH-RISK REQUEST DETECTED!{C_RESET}")
        print(f"Department: {C_YELLOW}{state.values.get('current_department')}{C_RESET}")
        print(f"Risk Category: {C_RED}{state.values.get('high_risk_type').upper()}{C_RESET}")
        print(f"\n{C_CYAN}--- Proposed Response (AI Supervisor Polished) ---{C_RESET}")
        print(state.values.get('agent_response'))
        print(f"{C_CYAN}--------------------------------------------------{C_RESET}")
        
        # Prompt the human supervisor in the terminal
        print(f"\n{C_BOLD}Supervisor Action Required:{C_RESET}")
        print("  [1] Approve response as-is")
        print("  [2] Reject response (provide feedback to customer)")
        print("  [3] Modify response text manually")
        
        choice = input("\nEnter choice (1, 2, or 3, Default: 1): ").strip()
        if not choice:
            choice = "1"
            
        approval_status = "Approved"
        approval_reason = "Approved by human supervisor."
        final_response = state.values.get('agent_response')
        
        if choice == "2":
            approval_status = "Rejected"
            reason = input("Enter rejection reason: ").strip()
            if not reason:
                reason = "Request does not meet our policy criteria."
            approval_reason = reason
            final_response = f"Your request for a {state.values.get('high_risk_type')} has been reviewed by our supervisor and declined. Reason: {reason}"
        elif choice == "3":
            approval_status = "Approved"
            approval_reason = "Modified and approved by human supervisor."
            print("\nEnter the modified response below (press Enter when done):")
            modified = input("> ").strip()
            if modified:
                final_response = modified
        
        print(f"\n-> Applying Supervisor Decision: {C_GREEN if approval_status == 'Approved' else C_RED}{approval_status}{C_RESET}")
        
        # Update the state of the graph, simulating the output of human_approval node
        app.update_state(
            config,
            {
                "approval_status": approval_status,
                "approval_reason": approval_reason,
                "final_response": final_response
            },
            as_node="human_approval"
        )
        
        # Resume graph execution
        print("Resuming graph execution...")
        app.invoke(None, config)
        
    # 4. Get the final output
    final_state = app.get_state(config)
    response = final_state.values.get("final_response")
    dept = final_state.values.get("current_department")
    
    print(f"\n{C_GREEN}{C_BOLD}=== Final Response to Customer ==={C_RESET}")
    print(f"{C_BOLD}Routed Department:{C_RESET} {dept}")
    print(f"{C_BOLD}Response:{C_RESET} {response}")
    print(f"{C_GREEN}=================================={C_RESET}")
    return response

def run_automated_demo(app):
    """Runs the 5 demonstration queries sequentially on a single thread (sharing memory)."""
    print(f"\n{C_CYAN}=== Starting Automated Demonstration ==={C_RESET}")
    print("All queries will run on the same Thread (demo_thread_001) to demonstrate memory retention.")
    
    config = {"configurable": {"thread_id": "demo_thread_001"}}
    
    queries = [
        "What are the pricing plans available for your software?",
        "I forgot my account password.",
        "My application crashes whenever I upload a file.",
        "I need a refund for my annual subscription.",
        "What was my previous support issue?"
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n\n{C_CYAN}{C_BOLD}------------------------------------------------------------------------")
        print(f" DEMONSTRATION QUERY {i}/5")
        print(f"------------------------------------------------------------------------{C_RESET}")
        run_query(app, config, q, customer_name="David")
        input(f"\nPress Enter to proceed to the next query...")
        
    print(f"\n{C_GREEN}{C_BOLD}=== Automated Demonstration Completed! ==={C_RESET}")

def start_interactive_chat(app):
    """Starts an interactive session with the customer support chatbot."""
    print(f"\n{C_CYAN}=== Starting Interactive Chat Session ==={C_RESET}")
    customer_name = input("Enter your name (Default: David): ").strip()
    if not customer_name:
        customer_name = "David"
        
    thread_id = input("Enter conversation thread ID (Default: chat_101): ").strip()
    if not thread_id:
        thread_id = "chat_101"
        
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\nChat thread '{C_YELLOW}{thread_id}{C_RESET}' started for customer '{C_GREEN}{customer_name}{C_RESET}'.")
    print("Type 'exit' or 'back' to return to main menu.")
    
    while True:
        try:
            query = input(f"\n{C_BOLD}{customer_name}:{C_RESET} ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "back", "quit"]:
                break
                
            run_query(app, config, query, customer_name=customer_name)
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            break

def view_database_contents():
    """Prints the contents of customer_interactions table in memory.db."""
    print(f"\n{C_CYAN}=== SQLite database: customer_interactions ==={C_RESET}")
    if not os.path.exists(DB_PATH):
        print("Database file does not exist yet. Run some queries first.")
        return
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customer_interactions ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("No interaction logs found in the database.")
            return
            
        header = f"| {'ID':<3} | {'Thread ID':<15} | {'Customer':<10} | {'Dept':<12} | {'Query':<30} |"
        print(header)
        print("-" * len(header))
        for row in rows:
            # truncate query for display
            q_disp = row[3][:27] + "..." if len(row[3]) > 30 else row[3]
            print(f"| {row[0]:<3} | {row[1]:<15} | {row[2]:<10} | {row[4]:<12} | {q_disp:<30} |")
    except Exception as e:
        print("Error reading database:", e)

def main():
    # Initialize the database and custom schema
    init_db()
    
    # Build LangGraph workflow
    workflow = build_support_graph()
    
    # Run the application loop inside the checkpointer's context manager
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
        
        while True:
            print_banner()
            print("Select an option:")
            print(f" {C_GREEN}[1]{C_RESET} Run Automated Demonstration (5 Sample Queries)")
            print(f" {C_GREEN}[2]{C_RESET} Start Interactive Chat Session")
            print(f" {C_GREEN}[3]{C_RESET} View SQLite Database Log Table")
            print(f" {C_GREEN}[4]{C_RESET} Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                run_automated_demo(app)
            elif choice == "2":
                start_interactive_chat(app)
            elif choice == "3":
                view_database_contents()
            elif choice == "4":
                print("\nThank you for using the support system. Goodbye!")
                break
            else:
                print(f"{C_RED}Invalid option. Please choose 1, 2, 3, or 4.{C_RESET}")
            
            input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    # Enable color support on Windows Command Prompt if needed
    if os.name == 'nt':
        os.system('color')
    main()
