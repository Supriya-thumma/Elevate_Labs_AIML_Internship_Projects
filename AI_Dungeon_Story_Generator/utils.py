import os

def save_story(story, file_name="generated_story.txt"):
    if not os.path.exists("stories"):
        os.makedirs("stories")
    file_path = os.path.join("stories", file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(story)
    return file_path
