import os
import torch
import numpy as np
import faiss
import pickle
from PIL import Image
import torchvision.transforms.v2 as T

class RetrievalEngine:
    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.index = None
        self.image_paths = []

    def extract_vector(self, img_path):
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor)
        return vec.squeeze(0).cpu().numpy().astype('float32')

    def build_and_save_index(self, target_dir, index_file="index.faiss", paths_file="paths.pkl"):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_paths = [
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.lower().endswith(valid_exts)
        ]
        
        vectors = [self.extract_vector(p) for p in self.image_paths]
        vectors_np = np.array(vectors)

        # Inner Product on L2-normalized vectors calculates cosine similarity
        self.index = faiss.IndexFlatIP(vectors_np.shape[1])
        self.index.add(vectors_np)

        # Save to disk
        faiss.write_index(self.index, index_file)
        with open(paths_file, 'wb') as f:
            pickle.dump(self.image_paths, f)
        print(f"Indexed {len(self.image_paths)} images and saved to {index_file}.")

    def load_index(self, index_file="index.faiss", paths_file="paths.pkl"):
        self.index = faiss.read_index(index_file)
        with open(paths_file, 'rb') as f:
            self.image_paths = pickle.load(f)

    def search(self, query_img_path, top_k=5):
        query_vec = self.extract_vector(query_img_path).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.image_paths[idx], float(score)))
        return results