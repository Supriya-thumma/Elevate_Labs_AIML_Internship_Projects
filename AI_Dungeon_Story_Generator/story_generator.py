from transformers import pipeline

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")

def generate_story(prompt, genre, tone, temperature=0.7, max_length=150):
    full_prompt = f"Genre: {genre}\nTone: {tone}\n\n{prompt}"
    output = generator(full_prompt, max_length=max_length, temperature=temperature, do_sample=True)
    return output[0]['generated_text']
