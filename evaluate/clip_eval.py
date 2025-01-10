import torch
import clip
from PIL import Image
import os
import numpy as np
from datetime import datetime


def clipEval(env_name, task_description):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Define the directory containing frames and the text prompt
    frames_dir = "evaluate/frames/"
    text_prompt = f'{env_name}: {task_description}'
    
    text = clip.tokenize([text_prompt]).to(device)
    text_features = model.encode_text(text)

    # Loop through images in the directory
    image_scores = []
    initial_time = datetime.now()
    for frame_file in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_file)

        if frame_file.lower().endswith('.png'):  # Check if it's a png
            # Load and preprocess the image
            image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)

            # Encode the image
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

            # Compute similarity score
            similarity = (image_features @ text_features.T).item()
            image_scores.append(similarity)
    return image_scores, np.mean(image_scores)

