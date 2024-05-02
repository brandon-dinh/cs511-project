from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import csv
import os
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

LIMIT = 5000
image_folder = "VG_100K"
def get_image_paths():
    with open('filenames.txt', 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
        return lines
    
image_paths = get_image_paths()
embeddings = []
count = 0
i = 0
for j in image_paths:
  image_path = os.path.join(image_folder, j)
  try:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
      image_features = model.encode_image(image)
      embeddings.append(image_features.cpu().numpy())
    if count % 100 == 0:
      print("Done with images through", i)
  except FileNotFoundError:
     print()
  i += 1
  count += 1



# save_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # L2 normalization
np.save("image_embeddings.npy", embeddings)