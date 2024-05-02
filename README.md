# Constructing Query-Agnostic Semantic Indexes for Unstructured Data with Foundational Models
Instructions:

To run the system, please follow the TASTI installation procedures at https://github.com/stanford-futuredata/tasti, but make sure to replace the TASTI folder with ours. Then, run the QASI.py with your custom query. The script will output the precision and recall. Then, run `blip_emb.ipynb` to calculate the metrics for the baseline. 

Some additional dependencies are required:
- transformers
- datasets


Baseline:
1. Download Visual Genome dataset: https://homes.cs.washington.edu/~ranjay/visualgenome/index.html

2. Generate ground truths for Visual Genome using ground_truth_generator.ipynb

3. Generate image embeddings using image_embeddings.py

4. Run cosine similarity using clip_cosine_similarity.py
