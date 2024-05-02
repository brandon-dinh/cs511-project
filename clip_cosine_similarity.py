import faiss
from sklearn.metrics import f1_score
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import clip
import os
import csv
import torch
import time
def encode_text(queries):
    """Encodes text queries using CLIP."""
    text_input = clip.tokenize(queries).to(device="cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        text_features = model.encode_text(text_input)

    encoded_queries = {}
    for i, query in enumerate(queries):
        text_features[i] = text_features[i] / np.linalg.norm(text_features[i], keepdims=True)
        encoded_queries[query] = text_features[i].cpu().numpy()
    return encoded_queries

def encode_images(image_folder, limit):
    embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_paths = []
    for i in range(limit):
        image_path = os.path.join(image_folder, f"{i}.jpg")
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_paths.append(image_path)
            with torch.no_grad():
                image_features = model.encode_image(image)
                embeddings.append(image_features.cpu().numpy())
                if i % 256 == 0:
                    print("Done with images through", i)
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
    return np.stack(embeddings), image_paths

def build_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def retrieve_images(index, query_features, paths, threshold=0.3):
    """Retrieves images similar to a query based on cosine similarity."""
    retrieved_images = []
    for key in query_features:
        D, I = index.search(query_features[key].reshape(1, -1), 10)
        for i in range(len(I[0])):
            print(D[0][i])
            if D[0][i] <= threshold:
                retrieved_images.append((I[0][i], paths[I[0][i]], key))
    return retrieved_images

def compute_f1(ground_truth, retrieved_images):
  """Computes F1 score based on ground truth and retrieved images."""
  predicted = [item[1] for item in retrieved_images]
  return f1_score(ground_truth, predicted, average='micro')

# Load CLIP model
model, preprocess = clip.load("ViT-B/32")

# Define paths and ground truth (replace with your data)
image_folder = "VG_100K"
queries = []
with open('questions.txt', 'r') as f:
    lines = f.readlines()
    queries = [line.strip() for line in lines]
# ground_truth = {"image1.jpg": "cat", "image2.png": "dog", "image3.jpeg": "bird"}
# path_to_image_id = {os.path.basename(path): ground_truth[path] for path in ground_truth}
print(queries)
# Encode data
text_features = encode_text(queries)

limit = 5000

# embeddings, image_paths = encode_images(image_folder, limit)

start_time = time.time()

embeddings = np.load('image_embeddings.npy').reshape(5000, 512)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# image_paths = []
# i = 0
# count = 0
# while i >= 0:
#     if count >= limit:
#         break
#     image_path = os.path.join(image_folder, f"{i}.jpg")
#     try:
#         image = Image.open(image_path)
#         image_paths.append(image_path)
#         i += 1
#         count += 1
#     except:
#         print(f"Error: Image not found at {image_path}")
#         i += 1

def get_image_paths():
    with open('filenames.txt', 'r') as file:
        lines = [line.strip() for line in file]
        return lines
    
image_paths = get_image_paths()
# Build index
index = build_index(embeddings)

# Retrieve images
retrieved_images = retrieve_images(index, text_features, image_paths)

# Compute F1 score
# f1 = compute_f1(list(ground_truth.values()), retrieved_images)

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

output_file = "result.txt"
with open(output_file, "w") as f:
    for i in retrieved_images:
        f.write(i[2] + ": " + i[1] + "\n")
# print(f"Retrieved images: {retrieved_images}")
# print(f"F1 score: {f1}")
