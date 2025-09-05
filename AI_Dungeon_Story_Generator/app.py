import streamlit as st
from story_generator import generate_story
import os

st.set_page_config(page_title="AI Dungeon Story Generator", layout="wide")

st.title("📚 AI Dungeon Story Generator")
st.write("Create interactive fantasy stories using AI GPT-2 (runs locally).")

# Sidebar
st.sidebar.header("Story Settings")
genre = st.sidebar.selectbox("Select Genre", ["Fantasy", "Mystery", "Sci-Fi", "Horror"])
tone = st.sidebar.selectbox("Select Tone", ["Dark", "Humorous", "Romantic", "Epic"])
temperature = st.sidebar.slider("Creativity (Temperature)", 0.5, 1.0, 0.7, 0.05)
max_length = st.sidebar.slider("Max Length", 50, 300, 150, 10)

# User input
st.subheader("Enter Your Story Prompt")
user_prompt = st.text_area("Start your story here...", height=150)

if st.button("Generate Story"):
    if user_prompt.strip():
        with st.spinner("Generating your story..."):
            story = generate_story(user_prompt, genre, tone, temperature, max_length)

        st.subheader("Your Story:")
        st.write(story)

        # Save story locally
        if not os.path.exists("stories"):
            os.makedirs("stories")
        file_path = os.path.join("stories", "generated_story.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(story)

        st.success(f"Story saved to {file_path}")
        st.download_button("Download Story as TXT", data=story, file_name="story.txt", mime="text/plain")
    else:
        st.warning("Please enter a story prompt before generating.")
