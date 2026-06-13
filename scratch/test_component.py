import streamlit as st
import streamlit.components.v1 as components
import os

st.title("Streamlit Bidirectional Component Test")

# Create a temporary directory for the component frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "test_frontend")
os.makedirs(frontend_dir, exist_ok=True)

html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Component</title>
</head>
<body style="background-color: #f0f2f6; font-family: sans-serif; padding: 10px;">
    <h3>Custom Component HTML</h3>
    <input type="text" id="my-input" value="Hello from JS!" style="padding: 5px; width: 200px;">
    <button id="my-btn" style="padding: 5px 10px;">Send to Python</button>
    <script>
        const btn = document.getElementById("my-btn");
        const input = document.getElementById("my-input");
        
        btn.addEventListener("click", () => {
            const val = input.value;
            // Send the message to Streamlit
            window.parent.postMessage({
                isStreamlitMessage: true,
                type: "streamlit:setComponentValue",
                value: val
            }, "*");
        });
        
        // Notify Streamlit that the component is ready
        window.parent.postMessage({
            isStreamlitMessage: true,
            type: "streamlit:componentReady",
            apiVersion: 1
        }, "*");
    </script>
</body>
</html>
"""

with open(os.path.join(frontend_dir, "index.html"), "w") as f:
    f.write(html_content)

# Declare component
my_component = components.declare_component("my_test_component", path=frontend_dir)

# Call component
res = my_component()

st.write("Value received from component:", res)
