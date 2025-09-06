from transformers import pipeline, set_seed as hf_set_seed
import re

# Load GPT-Neo 125M for better story generation but still lightweight
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=-1)  # CPU

hf_set_seed(42)  # default seed for reproducibility

# Simple list of harsh words to filter (expand as needed)
HARSH_WORDS = ["damn", "hell", "shit", "fuck", "bitch", "asshole", "crap"]

def soft_censor(text):
    def censor_word(match):
        word = match.group()
        return word[0] + "*" * (len(word) - 2) + word[-1] if len(word) > 2 else "*" * len(word)
    pattern = re.compile(r'\b(' + '|'.join(HARSH_WORDS) + r')\b', re.IGNORECASE)
    return pattern.sub(censor_word, text)

def generate_story(prompt, genre, tone, temperature=0.7, max_length=250, num_return_sequences=3):
    """
    Generate story continuations based on user prompt, genre, and tone.
    The prompt is combined with a guiding instruction to adapt style.
    """
    instruction = (
        f"Write a {genre.lower()} story with a {tone.lower()} tone. "
        "Continue the story below:\n\n"
    )
    
    full_prompt = instruction + prompt.strip() + "\n"

    outputs = generator(
        full_prompt,
        max_length=max_length + len(full_prompt.split()),  # add prompt length to max_length
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_return_sequences,
        pad_token_id=50256,  # EOS token for GPT-2/GPT-Neo
        repetition_penalty=1.2  # reduce repetition
    )

    stories = []
    for output in outputs:
        text = output['generated_text']
        continuation = text[len(full_prompt):].strip()
        if "\n\n" in continuation:
            continuation = continuation.split("\n\n")[0].strip()
        continuation = soft_censor(continuation)
        stories.append(continuation)

    return stories

def set_seed(seed):
    hf_set_seed(seed)
