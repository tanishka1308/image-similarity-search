import os
import json
import shutil
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as T

from src.model import ImageEmbeddingNet

def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class DynamicGroupManager:
    def __init__(self, origin_dir="dataset/images", base_output="sorted_folders", 
                 model_path="model_weights.pth", registry_file="registry.json"):
        self.origin_dir = origin_dir
        self.base_output = base_output
        self.registry_file = registry_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ImageEmbeddingNet(embedding_dim=256)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_transform()
        self.registry = self._load_or_build_registry()

    def _load_or_build_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                return json.load(f)
        
        registry = {"images": {}, "next_group_id": 1}
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        if os.path.exists(self.origin_dir):
            for fname in os.listdir(self.origin_dir):
                if fname.lower().endswith(valid_exts):
                    registry["images"][fname] = {
                        "current_folder": "origin",
                        "path": os.path.join(self.origin_dir, fname)
                    }
        self._save_registry(registry)
        return registry

    def _save_registry(self, registry=None):
        reg = registry if registry else self.registry
        with open(self.registry_file, "w") as f:
            json.dump(reg, f, indent=2)

    def extract_vector(self, img_path):
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor)
        return vec.squeeze(0).cpu().numpy().astype('float32')

    def process_query(self, query_path, threshold=0.70):
        query_fname = os.path.basename(query_path)
        img_info = self.registry["images"].get(query_fname)
        
        if not img_info:
            current_folder = "unknown"
            actual_path = query_path
        else:
            current_folder = img_info["current_folder"]
            actual_path = img_info["path"]

        if not os.path.exists(actual_path):
            print(f"[ERROR] Query file '{actual_path}' does not exist on disk.")
            return

        query_vec = self.extract_vector(actual_path)

        # -------------------------------------------------------------
        # CASE 1: Query belongs to an EXISTING group folder
        # -------------------------------------------------------------
        if current_folder not in ["origin", "unknown"]:
            print(f"\n=======================================================")
            print(f"[STATUS] Image '{query_fname}' already belongs to: '{current_folder}'")
            print(f"[INFO] Showing current members of '{current_folder}':")
            print(f"=======================================================\n")
            
            existing_members = [
                (fname, info["path"])
                for fname, info in self.registry["images"].items()
                if info["current_folder"] == current_folder
            ]
            
            for fname, path in existing_members:
                vec = self.extract_vector(path)
                sim = float(np.dot(query_vec, vec))
                print(f" • [Current Member] {fname} | Similarity: {sim:.4f}")

            print(f"\n[SCANNING ORIGIN] Checking if new matches exist in origin...")
            origin_candidates = [
                (fname, info["path"])
                for fname, info in self.registry["images"].items()
                if info["current_folder"] == "origin"
            ]
            print(f"Found {len(origin_candidates)} images remaining in origin to compare.")

            new_origin_matches = []
            for fname, path in origin_candidates:
                cand_vec = self.extract_vector(path)
                sim = float(np.dot(query_vec, cand_vec))
                print(f"Comparing with {fname}: similarity = {sim:.4f}")
                if sim >= threshold:
                    new_origin_matches.append((fname, path, sim))

            if len(new_origin_matches) == 0:
                print(f"-> No new matches >= {threshold:.2f} found in origin.")
            else:
                target_dir = os.path.join(self.base_output, current_folder)
                print(f"-> Adding {len(new_origin_matches)} matching image(s) to '{current_folder}':\n")
                for fname, old_path, sim in new_origin_matches:
                    new_path = os.path.join(target_dir, fname)
                    shutil.move(old_path, new_path)
                    self.registry["images"][fname] = {
                        "current_folder": current_folder,
                        "path": new_path
                    }
                    print(f" • [Moved to {current_folder}] {fname} (Score: {sim:.4f})")
                self._save_registry()

            remaining = sum(1 for info in self.registry["images"].values() if info["current_folder"] == "origin")
            print(f"\nTotal images left in origin: {remaining}")
            return

        # -------------------------------------------------------------
        # CASE 2: Query is in ORIGIN -> Form a BRAND NEW group folder
        # -------------------------------------------------------------
        print(f"\n=======================================================")
        print(f"[STATUS] Query image '{query_fname}' is in 'origin'. Creating new group...")
        print(f"=======================================================\n")

        origin_candidates = [
            (fname, info["path"])
            for fname, info in self.registry["images"].items()
            if info["current_folder"] == "origin"
        ]

        print(f"Total candidate images found in origin: {len(origin_candidates)}")

        matched_files = []
        for fname, path in origin_candidates:
            cand_vec = self.extract_vector(path)
            sim = float(np.dot(query_vec, cand_vec))
            print(f"Comparing with {fname}: similarity = {sim:.4f}")
            if sim >= threshold:
                matched_files.append((fname, path, sim))

        if len(matched_files) == 0:
            print(f"No images in origin reached threshold >= {threshold:.2f}")
            return

        matched_files.sort(key=lambda x: x[2], reverse=True)

        group_id = self.registry["next_group_id"]
        group_name = f"group_{group_id:03d}"
        target_dir = os.path.join(self.base_output, group_name)
        os.makedirs(target_dir, exist_ok=True)

        print(f"\nCreated folder: '{target_dir}/'")
        print(f"Moving {len(matched_files)} matching images into {group_name}:\n")

        for fname, old_path, sim in matched_files:
            new_path = os.path.join(target_dir, fname)
            shutil.move(old_path, new_path)
            self.registry["images"][fname] = {
                "current_folder": group_name,
                "path": new_path
            }
            print(f" • [Moved] {fname} (Score: {sim:.4f})")

        self.registry["next_group_id"] += 1
        self._save_registry()

        remaining = sum(1 for info in self.registry["images"].values() if info["current_folder"] == "origin")
        print(f"\n[DONE] Group '{group_name}' formed. Total images remaining in origin: {remaining}")

def main():
    parser = argparse.ArgumentParser(description="Dynamic grouping manager with origin replenishment.")
    parser.add_argument("--query", type=str, required=True, help="Filename or path of query image")
    parser.add_argument("--origin", type=str, default="dataset/images", help="Origin directory")
    parser.add_argument("--output", type=str, default="sorted_folders", help="Base directory for groups")
    parser.add_argument("--threshold", type=float, default=0.50, help="Similarity cutoff threshold (0.0 - 1.0)")
    args = parser.parse_args()

    manager = DynamicGroupManager(origin_dir=args.origin, base_output=args.output)
    manager.process_query(args.query, threshold=args.threshold)

if __name__ == "__main__":
    main()