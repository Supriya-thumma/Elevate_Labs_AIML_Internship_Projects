# app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv
from io import StringIO

# Load local .env if exists (for local dev)
load_dotenv()

# Constants
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
# You can change model to "gpt2-medium" or others if you want, but gpt2 is default

# Get Hugging Face API token from Streamlit secrets or environment variable
HF_API_TOKEN = st.secrets["HF_API_TOKEN"] if "HF_API_TOKEN" in st.secrets else os.getenv("HF_API_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Genres and tones
GENRES = ["Fantasy", "Mystery", "Sci-Fi", "Horror"]
TONES = ["Dark", "Humorous", "Romantic", "Epic"]

# Max characters for character traits input
MAX_CHAR_TRAITS = 200

def query_hf_api(prompt, temperature, max_tokens, n_variations):
    """
    Query Hugging Face GPT-2 Inference API for text generation.
    Returns list of generated texts or raises Exception on error.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "num_return_sequences": n_variations,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    data = response.json()
    # data is a list of dicts with 'generated_text' keys
    if not isinstance(data, list):
        raise Exception(f"Unexpected API response format: {data}")
    return [item["generated_text"].strip() for item in data]

def build_prompt(genre, tone, characters, user_prompt):
    """
    Build enhanced prompt by prepending genre, tone, and character info.
    """
    char_desc = ""
    if characters:
        char_lines = []
        for i, c in enumerate(characters, 1):
            name = c.get("name", "").strip()
            role = c.get("role", "").strip()
            traits = c.get("traits", "").strip()
            desc = f"{name} ({role})"
            if traits:
                desc += f", traits: {traits}"
            char_lines.append(desc)
        char_desc = "Characters:\n" + "\n".join(char_lines) + "\n\n"
    prompt = (
        f"Genre: {genre}\n"
        f"Tone: {tone}\n\n"
        f"{char_desc}"
        f"Story start:\n{user_prompt.strip()}\n\nContinue the story:"
    )
    return prompt

def generate_title(story_text):
    """
    Generate a short creative title for the story using GPT-2.
    We will prompt GPT-2 with "Title:" after the story text.
    """
    prompt = story_text.strip() + "\n\nTitle:"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 10,
            "num_return_sequences": 1,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "return_full_text": False,
            "stop": ["\n"]
        },
        "options": {"wait_for_model": True},
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
    if response.status_code != 200:
        return "Untitled Story"
    data = response.json()
    if not isinstance(data, list) or not data:
        return "Untitled Story"
    title = data[0].get("generated_text", "").strip()
    # Clean title: remove trailing punctuation or newlines
    title = title.split("\n")[0].strip(" .,:;-")
    if not title:
        title = "Untitled Story"
    return title

def save_story_to_txt(title, prompt, continuations):
    """
    Create a .txt content string with title, prompt, and all continuations.
    """
    lines = [f"Title: {title}", "", "Prompt:", prompt, "", "Story Continuations:"]
    for i, cont in enumerate(continuations, 1):
        lines.append(f"\n--- Variation {i} ---\n{cont}")
    return "\n".join(lines)

def main():
    st.set_page_config(page_title="AI Dungeon Story Generator", layout="wide")

    st.title("📝 AI Dungeon Story Generator")
    st.markdown(
        """
        Generate creative story continuations using GPT-2 via Hugging Face Inference API.
        Customize genre, tone, characters, and generation parameters.
        """
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Story Settings")

        genre = st.selectbox("Select Genre", GENRES, index=0)
        tone = st.selectbox("Select Tone", TONES, index=3)

        st.markdown("### Character Customization")
        # Allow up to 3 characters
        characters = []
        for i in range(1, 4):
            with st.expander(f"Character {i} (optional)"):
                name = st.text_input(f"Name {i}", key=f"name_{i}")
                role = st.text_input(f"Role {i}", key=f"role_{i}")
                traits = st.text_area(f"Traits {i} (comma separated)", max_chars=MAX_CHAR_TRAITS, key=f"traits_{i}")
                if name or role or traits:
                    characters.append({"name": name, "role": role, "traits": traits})

        st.markdown("---")
        st.header("Generation Parameters")
        temperature = st.slider("Temperature (Creativity)", 0.1, 1.0, 0.7, 0.05)
        max_tokens = st.slider("Max Tokens (Length)", 50, 300, 150, 10)
        n_variations = st.slider("Number of Variations", 1, 3, 1)

        st.markdown("---")
        # Buttons
        generate_btn = st.button("Generate Story Continuation")
        new_story_btn = st.button("Start New Story (Clear History)")

        st.markdown("---")
        # Download buttons will be shown in main area after generation

    # Session state initialization
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {prompt, genre, tone, characters, continuations, title}
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "current_title" not in st.session_state:
        st.session_state.current_title = ""
    if "current_continuations" not in st.session_state:
        st.session_state.current_continuations = []

    # Start new story clears history and current data
    if new_story_btn:
        st.session_state.history = []
        st.session_state.current_prompt = ""
        st.session_state.current_title = ""
        st.session_state.current_continuations = []
        st.experimental_rerun()

    # User prompt input
    st.markdown("## Enter Your Story Starting Text")
    user_prompt = st.text_area("Your prompt:", value=st.session_state.current_prompt, height=150)

    # Update current prompt in session state
    st.session_state.current_prompt = user_prompt

    # Generate story continuation
    if generate_btn:
        if not HF_API_TOKEN:
            st.error("Hugging Face API token not found. Please set it in Streamlit secrets or .env file as HF_API_TOKEN.")
            st.stop()
        if not user_prompt.strip():
            st.warning("Please enter a starting prompt to generate story continuation.")
            st.stop()

        prompt = build_prompt(genre, tone, characters, user_prompt)
        with st.spinner("Generating story continuation..."):
            try:
                continuations = query_hf_api(prompt, temperature, max_tokens, n_variations)
            except Exception as e:
                st.error(f"Error during generation: {e}")
                st.stop()

        # Generate title from first continuation + prompt
        try:
            title = generate_title(user_prompt + "\n\n" + continuations[0])
        except Exception:
            title = "Untitled Story"

        # Save current generation to session state
        st.session_state.current_title = title
        st.session_state.current_continuations = continuations

        # Append to history
        st.session_state.history.append(
            {
                "prompt": user_prompt,
                "genre": genre,
                "tone": tone,
                "characters": characters,
                "continuations": continuations,
                "title": title,
            }
        )

    # Display story history and current generation
    if st.session_state.history:
        st.markdown("## Story History")
        for idx, entry in enumerate(st.session_state.history):
            st.markdown(f"### Story {idx + 1}: **{entry['title']}**")
            st.markdown(f"**Genre:** {entry['genre']} | **Tone:** {entry['tone']}")
            if entry["characters"]:
                char_lines = []
                for c in entry["characters"]:
                    line = f"- {c.get('name','')} ({c.get('role','')})"
                    traits = c.get("traits", "").strip()
                    if traits:
                        line += f", traits: {traits}"
                    char_lines.append(line)
                st.markdown("**Characters:**\n" + "\n".join(char_lines))
            st.markdown("**Prompt:**")
            st.write(entry["prompt"])
            st.markdown("**Continuations:**")
            for i, cont in enumerate(entry["continuations"], 1):
                st.markdown(f"**Variation {i}:**")
                st.write(cont)
            st.markdown("---")

    # Download buttons for current story
    if st.session_state.current_continuations:
        st.markdown("## Download Current Story")

        # Download single variation buttons
        for i, cont in enumerate(st.session_state.current_continuations, 1):
            filename = f"{st.session_state.current_title.replace(' ', '_')}_variation_{i}.txt"
            content = f"Title: {st.session_state.current_title}\n\nPrompt:\n{st.session_state.current_prompt}\n\nContinuation:\n{cont}"
            st.download_button(
                label=f"Download Variation {i} as .txt",
                data=content,
                file_name=filename,
                mime="text/plain",
                key=f"download_var_{i}",
            )

        # Download all variations in one file
        all_content = save_story_to_txt(
            st.session_state.current_title,
            st.session_state.current_prompt,
            st.session_state.current_continuations,
        )
        all_filename = f"{st.session_state.current_title.replace(' ', '_')}_all_variations.txt"
        st.download_button(
            label="Download All Variations as Single .txt",
            data=all_content,
            file_name=all_filename,
            mime="text/plain",
            key="download_all",
        )

    # Footer / instructions
    st.markdown("---")
    st.markdown(
        """
        <small>
        **Note:** This app uses the Hugging Face GPT-2 model via the Inference API.<br>
        You must provide your Hugging Face API token as `HF_API_TOKEN` in Streamlit secrets or a `.env` file.<br>
        For local development, create a `.env` file with:<br>
        <code>HF_API_TOKEN=your_token_here</code><br>
        For Streamlit Cloud deployment, add the token in the Secrets section.<br>
        </small>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()