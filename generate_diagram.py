from src.workflow import build_support_graph

try:
    workflow = build_support_graph()
    app = workflow.compile()
    
    # Try drawing mermaid PNG
    # This might require pyppeteer or an internet connection
    png_data = app.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(png_data)
    print("Success: Generated workflow.png using draw_mermaid_png()")
except Exception as e:
    print("Error using draw_mermaid_png:", e)
    
    # Fallback: Save a Mermaid markdown file so the user can easily render it
    try:
        mermaid_text = app.get_graph().draw_mermaid()
        with open("workflow.mmd", "w") as f:
            f.write(mermaid_text)
        print("Success: Generated workflow.mmd (Mermaid Markdown fallback)")
    except Exception as e2:
        print("Error saving Mermaid text:", e2)
