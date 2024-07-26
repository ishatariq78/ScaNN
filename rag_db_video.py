import scann
import numpy as np
import torch

# Load embeddings
embeddings = torch.load('/lustre/fs1/home/itariq/VideoCrafter/data/embed_vidcap1.pt')
video_embeddings = embeddings['videos']

video_embedding_0 = video_embeddings[0]
print("video 0th embedding shape", video_embedding_0.shape)


stacked_embeddings = []

# Iterate over each video embedding
for video_embedding in video_embeddings:
    # Calculate the average embedding across frames
    averaged_embedding = np.mean(video_embedding, axis=0, keepdims=True)
    # Append to the list of stacked embeddings
    stacked_embeddings.append(averaged_embedding)

# Stack all embeddings into a single numpy array
video_embeddings_stacked = np.vstack(stacked_embeddings)
print("stacked emb shapes", video_embeddings_stacked.shape)


# Build ScaNN index
print("going to build scann index")
searcher = scann.scann_ops_pybind.builder(video_embeddings_stacked, 10, "dot_product").tree(
    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

# Save ScaNN index
searcher.serialize("/lustre/fs1/home/itariq/VideoCrafter/data/scann_updated/")
