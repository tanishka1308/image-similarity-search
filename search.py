import os
import argparse
import torch
from src.model import ImageEmbeddingNet
from src.engine import RetrievalEngine

def main():
    parser = argparse.ArgumentParser(description="Image Similarity Search")
    parser.add_argument("--query", type=str, required=True, help="Path to query image")
    parser.add_argument("--index", action="store_true", help="Re-index the directory before search")
    parser.add_argument("--dir", type=str, default="dataset", help="Target directory to search inside")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar images to return")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageEmbeddingNet(embedding_dim=256)
    
    # Load trained weights if available
    if os.path.exists("model_weights.pth"):
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
        print("Loaded trained model_weights.pth")
    else:
        print("No fine-tuned weights found. Using default pretrained weights.")

    engine = RetrievalEngine(model, device=device)

    # Build or load FAISS index
    if args.index or not os.path.exists("index.faiss"):
        print("Building vector index...")
        engine.build_and_save_index(args.dir)
    else:
        engine.load_index()

    # Search
    print(f"\nQuerying: {args.query}")
    matches = engine.search(args.query, top_k=args.top_k)
    print("\n--- Top Matches ---")
    for rank, (path, score) in enumerate(matches, 1):
        print(f"{rank}. {path} | Similarity: {score:.4f}")

if __name__ == "__main__":
    main()