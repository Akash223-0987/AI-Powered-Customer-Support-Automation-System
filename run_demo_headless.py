import os
from unittest.mock import patch
import main

# Define the sequence of inputs to mock:
# 1. Choose Option 1 (Run Automated Demonstration) from main menu
# 2. Press Enter to proceed after Query 1
# 3. Press Enter to proceed after Query 2
# 4. Press Enter to proceed after Query 3
# 5. At the Human Supervisor prompt for Query 4 (Refund), choose "1" (Approve)
# 6. Press Enter to proceed after Query 4
# 7. Press Enter to proceed after Query 5 (Memory Recall)
# 8. Choose Option 4 (Exit) from the main menu
mocked_inputs = [
    "1",  # main menu choice
    "",   # advance query 1
    "",   # advance query 2
    "",   # advance query 3
    "1",  # approve query 4 refund
    "",   # advance query 4
    "",   # advance query 5
    "4"   # exit main menu
]

input_iterator = iter(mocked_inputs)

def mock_input(prompt=""):
    print(f"\n{main.C_YELLOW}[MOCK INPUT]{main.C_RESET} {prompt}", end="")
    try:
        val = next(input_iterator)
        print(f"{main.C_BOLD}{val}{main.C_RESET}")
        return val
    except StopIteration:
        print("4")
        return "4"

def run_headless():
    # Force ANSI color codes on Windows
    if os.name == 'nt':
        os.system('color')
        
    print(f"{main.C_CYAN}=====================================================")
    print("Running Customer Support Automation System Headlessly")
    print(f"====================================================={main.C_RESET}\n")
    
    with patch('builtins.input', mock_input):
        main.main()

if __name__ == "__main__":
    run_headless()
