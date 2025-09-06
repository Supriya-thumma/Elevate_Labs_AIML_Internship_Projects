import streamlit as st
from story_generator import generate_story, set_seed
import os
from datetime import datetime
import random
import shutil

st.set_page_config(page_title="AI Dungeon Story Generator", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    color: #333333;
}
h1, h2, h3 {
    font-family: 'Georgia', serif;
}
.story-box {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 15px;
    white-space: pre-wrap;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(to bottom, #6a11cb, #2575fc);
    color: white;
}
footer {
    text-align: center;
    font-size: 0.8em;
    color: #888888;
    margin-top: 20px;
}
.sample-prompt-btn {
    margin-bottom: 5px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 AI Dungeon Story Generator")
st.write("Create interactive fantasy stories using AI GPT-Neo (runs locally).")

# Sample prompts for user guidance
SAMPLE_PROMPTS = [
    "In a dark forest, a mysterious figure appears under the full moon...",
    "The ancient artifact was said to grant any wish, but at a terrible cost.",
    "Detective Marla stepped into the abandoned mansion, her flashlight flickering.",
    "On the distant planet of Zorax, the first human colony faced an unexpected threat.",
    "A young wizard discovers a hidden power that could change the fate of the kingdom."
]

# Sidebar for settings and samples
st.sidebar.header("Story Settings")
genre = st.sidebar.selectbox("Select Genre", ["Fantasy", "Mystery", "Sci-Fi", "Horror"])
tone = st.sidebar.selectbox("Select Tone", ["Dark", "Humorous", "Romantic", "Epic"])
temperature = st.sidebar.slider("Creativity (Temperature)", 0.5, 1.0, 0.7, 0.05)
max_length = st.sidebar.slider("Max Length", 100, 300, 250, 10)
num_stories = st.sidebar.slider("Number of Story Variations", 1, 5, 3)
character_name = st.sidebar.text_input("Character Name (optional)")
seed_input = st.sidebar.text_input("Random Seed (optional, integer)")

st.sidebar.markdown("---")
if st.sidebar.button("Clear All Saved Stories"):
    if os.path.exists("stories"):
        shutil.rmtree("stories")
    st.sidebar.success("All saved stories cleared.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Sample Prompts")
for i, sample in enumerate(SAMPLE_PROMPTS, 1):
    if st.sidebar.button(f"Use Sample {i}"):
        st.session_state['user_prompt'] = sample
        st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by AI Dungeon Team. Powered by GPT-Neo 125M.")

# Initialize session state for prompt and stories
if 'user_prompt' not in st.session_state:
    st.session_state['user_prompt'] = ""

if 'stories' not in st.session_state:
    st.session_state['stories'] = []

# User input prompt
st.subheader("Enter Your Story Prompt")
user_prompt = st.text_area("Start your story here...", height=150, key='user_prompt')

col1, col2 = st.columns([1, 1])
with col1:
    generate_btn = st.button("Generate Story")
with col2:
    clear_btn = st.button("Clear Prompt")

if clear_btn:
    st.session_state['user_prompt'] = ""
    st.session_state['stories'] = []
    st.experimental_rerun()

if generate_btn:
    if user_prompt.strip():
        # Set seed if provided
        if seed_input.strip().isdigit():
            seed_val = int(seed_input.strip())
            set_seed(seed_val)
        else:
            seed_val = random.randint(0, 10000)
            set_seed(seed_val)

        # Add character name to prompt if provided
        prompt_to_use = user_prompt
        if character_name.strip():
            prompt_to_use = f"{character_name.strip()} is the main character. {user_prompt.strip()}"

        with st.spinner("Generating your story... This may take a few seconds."):
            stories = generate_story(prompt_to_use, genre, tone, temperature, max_length, num_stories)

        st.session_state['stories'] = stories

# Display generated stories with continue option
if st.session_state['stories']:
    st.subheader("Your Story Variations:")
    for i, story in enumerate(st.session_state['stories'], 1):
        with st.expander(f"Variation {i}"):
            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                story_file_name = f"story_{datetime.now().strftime('%Y%m%d_%H%M%S')}_var{i}.txt"
                st.download_button(
                    label="Download this story",
                    data=story,
                    file_name=story_file_name,
                    mime="text/plain"
                )
            with col2:
                if st.button(f"Continue this story (Variation {i})"):
                    # Append selected story continuation to prompt for further generation
                    st.session_state['user_prompt'] = st.session_state['user_prompt'].strip() + "\n" + story.strip()
                    st.session_state['stories'] = []
                    st.experimental_rerun()

    # Save all stories with timestamp
    if st.button("Save All Stories"):
        if not os.path.exists("stories"):
            os.makedirs("stories")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("stories", f"generated_story_{timestamp}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Genre: {genre}\nTone: {tone}\nSeed: {seed_input if seed_input.strip().isdigit() else 'Random'}\nCharacter Name: {character_name}\n\nUser  Prompt:\n{user_prompt}\n\n")
            for i, story in enumerate(st.session_state['stories'], 1):
                f.write(f"--- Variation {i} ---\n{story}\n\n")
        st.success(f"Stories saved to {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            st.download_button("Download All Stories as TXT", data=f.read(), file_name=f"story_{timestamp}.txt", mime="text/plain")

# Footer
st.markdown("""
<footer>
    <hr>
    <p>© 2025 AI Dungeon Story Generator. All rights reserved.</p>
</footer>
""", unsafe_allow_html=True)
