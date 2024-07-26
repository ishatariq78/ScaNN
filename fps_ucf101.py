import os
import torch
from tqdm import tqdm
import av
from transformers import CLIPProcessor, CLIPModel

# Paths
main_videos_path = "/lustre/fs1/home/itariq/VideoCrafter/data/UCF-101/"

# Collect all video files from subdirectories
video_files = []
for root, _, files in os.walk(main_videos_path):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mkv')):  # Add video formats as needed
            video_files.append(os.path.join(root, file))

print("video file path", video_files[0])

# Load CLIP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = '/lustre/fs1/home/itariq/hf_models/openai/clip-vit-base-patch32'
processor = CLIPProcessor.from_pretrained(model_dir)
model = CLIPModel.from_pretrained(model_dir).to(device)

# Function to load and preprocess a video, limiting to 40 frames
def load_and_preprocess_video(video_path, max_frames=60):
    container = av.open(video_path)
    frames = []
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count >= max_frames:
            break
        try:
            img = frame.to_image()
            processed_img = processor(images=img, return_tensors="pt").pixel_values
            frames.append(processed_img)
            frame_count += 1
        except Exception as e:
            print(f"Error processing frame from {video_path}: {e}")
            continue

    # Concatenate frames along the batch dimension
    return torch.cat(frames, dim=0).to(device)

# Generate embeddings
video_embeddings = []

for video_path in tqdm(video_files):
    # Load and preprocess entire video
    video_tensor = load_and_preprocess_video(video_path)
    print("video tensor is", video_tensor.shape)
    
    # Encode video
    with torch.no_grad():
        video_embedding = model.get_image_features(video_tensor)
    video_embeddings.append(video_embedding.cpu().numpy())

# Save embeddings
torch.save({'videos': video_embeddings}, '/lustre/fs1/home/itariq/VideoCrafter/data/embed_ucf2.pt')
