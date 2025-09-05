import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env (for local development)
load_dotenv()

# App title
st.set_page_config(page_title="AI Dungeon Story Generator", page_icon="📚")
st.title("📚 AI Dungeon Story Generator")
st.markdown("Create interactive fantasy stories using AI.")

# Load API token from Streamlit secrets or .env
HF_API_TOKEN = st.secrets["HF_API_TOKEN"] if "HF_API_TOKEN" in st.secrets else os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    st.error("Hugging Face API token not found. Please add it in `.streamlit/secrets.toml` or `.env` as `HF_API_TOKEN`.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/gpt2"

# ✅ Debugging token validity
st.write("✅ Token starts with:", HF_API_TOKEN[:8])
test_response = requests.get("https://api-inference.huggingface.co/status/gpt2", headers=HEADERS)
st.write("✅ Token check status:", test_response.status_code)

if test_response.status_code == 401:
    st.error("❌ Invalid Hugging Face token. Please check your HF_API_TOKEN in secrets.toml or Hugging Face account.")
    st.stop()

# Story generator function
def generate_story(prompt, max_length=200):
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": max_length, "temperature": 0.9, "top_p": 0.95}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        return f"❌ Error during generation: {response.status_code} - {response.text}"
    
    data = response.json()
    return data[0]['generated_text']

# Streamlit UI
st.subheader("Enter Your Adventure Prompt")
user_prompt = st.text_area("Example: 'You wake up in a dark forest with no memory of how you got there...'")

max_length = st.slider("Story Length", 50, 500, 200)
generate_btn = st.button("Generate Story")

if generate_btn:
    if not user_prompt.strip():
        st.warning("Please enter a prompt to start the adventure.")
    else:
        with st.spinner("Generating your adventure..."):
            story = generate_story(user_prompt, max_length)
        st.subheader("✨ Your AI-Generated Adventure")
        st.write(story)
