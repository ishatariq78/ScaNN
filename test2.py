import numpy as np
import scann
import torch
import cv2
import os
from tqdm import tqdm
from PIL import Image
import av
from transformers import CLIPProcessor, CLIPModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'

embeddings = torch.load('/lustre/fs1/home/itariq/VideoCrafter/data/embed_ucf2.pt')
video_embeddings = embeddings['videos']

model_dir = '/lustre/fs1/home/itariq/hf_models/openai/clip-vit-base-patch32'
processor = CLIPProcessor.from_pretrained(model_dir)
model = CLIPModel.from_pretrained(model_dir).to(device)

#caption = "A dog catching a frisbee in a park" #prompt1
#caption = "A person executing a series of karate moves in a training room" #prompt2
caption = "A group of people enjoying a dance party with colorful lights" #prompt3


inputs = processor(text=caption, return_tensors="pt", truncation=True)
with torch.no_grad():
    text_embedding = model.get_text_features(input_ids=inputs.input_ids.to(device))

text_embed = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
text_embed = text_embedding.cpu().numpy().flatten()

# Load the ScaNN searcher from the saved path
searcher_path = "/lustre/fs1/home/itariq/VideoCrafter/data/scann_ucf101/"  # Replace with your actual path
searcher = scann.scann_ops_pybind.load_searcher(searcher_path)

# Perform the search
neighbors, distances = searcher.search(text_embed)
neighbors = np.array(neighbors, dtype=np.int64)
neighbors_tensor = torch.tensor(neighbors, dtype=torch.long, device=device)

# Retrieve the embeddings of the nearest neighbors
retrieved_embeddings = [video_embeddings[idx] for idx in neighbors_tensor.cpu().numpy()] 

# Print the results
print("Nearest neighbors (indices):", neighbors)
print("Distances:", distances)

videos_path = "/lustre/fs1/home/itariq/VideoCrafter/data/UCF-101/"
output_dir = '/lustre/fs1/home/itariq/VideoCrafter/data/output_videos'
os.makedirs(output_dir, exist_ok=True)

# Parse info.txt
video_files = []
for root, _, files in os.walk(videos_path):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mkv')):  # Add video formats as needed
            video_files.append(os.path.join(root, file))

'''
# Function to resize video
def resize_video(input_path, output_path, width, height):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)

    cap.release()
    out.release()

# Desired resolution
desired_width = 640
desired_height = 480

# Resize and save the top 9 nearest neighbor videos
for i in range(10):
    nearest_video_path = video_files[neighbors[i]]
    output_video_path = os.path.join(output_dir, f"nearest_video_{i}.avi")
    resize_video(nearest_video_path, output_video_path, desired_width, desired_height)
    print(f"Nearest neighbor video {i} resized and saved to: {output_video_path}")
'''
for i in range(10):
    nearest_video_path = video_files[neighbors[i]]
    print(f"Nearest neighbor video path: {nearest_video_path}")