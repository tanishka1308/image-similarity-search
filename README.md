# Content-Based Image Retrieval System (CBIR)

A deep learning image retrieval system built with PyTorch, ResNet-50, and FAISS. Given a query image, the system identifies and ranks the most visually and semantically similar images in the directory.

## Features
- **Feature Backbone**: Pretrained ResNet-50 with a custom projection head (256-D metric space).
- **Metric Learning**: Self-supervised training using Triplet Margin Loss with on-the-fly synthetic distortions (perspective warp, color jitter, sensor noise, random erasing).
- **Fast Similarity Search**: Offline indexing and cosine nearest-neighbor search powered by FAISS.

## Project Structure
```text
image_retrieval_system/
├── dataset/             # Target images for indexing
├── src/
│   ├── model.py         # ResNet-50 + Projection Head
│   ├── dataset.py       # Triplet data pipeline with synthetic augmentations
│   └── engine.py        # FAISS indexing & retrieval logic
├── train.py             # Metric learning training script
├── search.py            # CLI retrieval script
└── requirements.txt
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Index and search:**
   ```bash
   python search.py --query dataset/images/sample.jpg --dir dataset/images --index --top_k 5
   ```