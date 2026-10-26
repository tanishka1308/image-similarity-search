# 🧠 AI-Powered Image Similarity Search & Dynamic Grouping

A deep-learning-based **Content-Based Image Retrieval (CBIR)** system that identifies visually similar images using computer vision, deep feature embeddings, and vector similarity search.

The project uses a pretrained **ResNet-50** backbone with a custom embedding layer to represent images as 256-dimensional vectors. It supports image similarity search, interactive image grouping, and a visual dashboard built with Streamlit.

🔗 **GitHub Repository:** [image-similarity-search](https://github.com/tanishka1308/image-similarity-search)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [How It Works](#-how-it-works)
- [Project Architecture](#-project-architecture)
- [Prerequisites](#-prerequisites)
- [Installation and Setup](#-installation-and-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Training the Model](#-training-the-model)
- [Running the Application](#-running-the-application)
- [Using the Image Search CLI](#-using-the-image-search-cli)
- [Similarity Score Explained](#-similarity-score-explained)
- [Configuration and Generated Files](#-configuration-and-generated-files)
- [Troubleshooting](#-troubleshooting)
- [Applications and Use Cases](#-applications-and-use-cases)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Overview

Finding visually similar images in a large collection can be time-consuming when done manually.

This project addresses that problem through deep-learning-based image representation and similarity matching.

Given a query image, the system extracts its visual features, converts them into a normalized embedding vector, and compares that representation with other images.

The system also includes a dynamic grouping interface that helps users organize related images into clusters based on a configurable similarity threshold.

### 🎯 Project Objectives

- Automate visual image retrieval.
- Generate meaningful image embeddings using deep learning.
- Rank images according to their visual similarity.
- Accelerate similarity searches using FAISS.
- Organize related images through dynamic grouping.
- Provide an interactive dashboard for exploring image collections.

---

## ✨ Key Features

### 1. Deep Learning-Based Feature Extraction

Uses a pretrained ResNet-50 convolutional neural network as the feature-extraction backbone.

A custom projection head transforms the extracted features into a **256-dimensional embedding space**.

### 2. Metric Learning

Implements triplet-based metric learning using PyTorch's `TripletMarginLoss`.

Training samples consist of:

- **Anchor:** The original image.
- **Positive:** An augmented version of the anchor image.
- **Negative:** A different image selected from the dataset.

The training objective encourages the anchor and positive embeddings to become closer than the anchor and negative embeddings.

### 3. Image Similarity Search

Accepts a query image and retrieves the closest matching images from a collection.

The retrieval engine uses normalized embeddings and cosine similarity to rank the results.

### 4. FAISS-Powered Vector Search

Uses FAISS to index image embeddings and efficiently retrieve nearest neighbors.

The index and corresponding image paths can be saved locally for subsequent searches.

### 5. Interactive Streamlit Dashboard

Provides a visual interface for interacting with the image retrieval and grouping workflow.

Dashboard capabilities include:

- Selecting an existing image from the collection.
- Uploading an image directly.
- Running similarity searches.
- Adjusting the similarity threshold.
- Exploring dynamically grouped images.
- Viewing image collections through horizontal carousels.
- Monitoring image and cluster counts.
- Switching between visual themes.
- Resetting the grouping workflow.

### 6. Dynamic Image Grouping

Organizes related images into groups according to their similarity.

The interface includes an origin image pool and grouped collections, allowing users to explore and manage the organization of their images.

### 7. Multiple Visual Themes

The dashboard supports three themes:

- Cyber Dark
- Midnight Purple
- Clean Light Glass

### 8. CPU and GPU Support

The model selects CUDA when a compatible GPU is available and otherwise runs on the CPU.

GPU availability depends on your installed PyTorch build and hardware.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning and model training |
| Torchvision | ResNet-50 and image transformations |
| ResNet-50 | Image feature extraction |
| FAISS | Vector indexing and nearest-neighbor search |
| NumPy | Numerical operations and embedding arrays |
| Pillow (PIL) | Image loading and processing |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive data visualizations |
| Triplet Margin Loss | Metric-learning objective |
| JSON | Image registry and grouping metadata |
| Pickle | Local storage of indexed image paths |

---

## ⚙️ How It Works

The system follows a pipeline that converts images into numerical representations and uses those representations to identify similar images.

### Step 1: Image Input

Images are loaded from the dataset directory or supplied by the user as query images.

Supported dataset image formats include:

- JPG / JPEG
- PNG
- BMP
- WEBP

### Step 2: Image Preprocessing

Images are converted to RGB, resized to 224 × 224 pixels, converted to tensors, and normalized using the standard ImageNet channel statistics.

### Step 3: Feature Extraction

The pretrained ResNet-50 backbone extracts visual features from the processed image.

### Step 4: Embedding Generation

A custom projection head maps the extracted features into a 256-dimensional embedding.

The resulting embedding is L2-normalized to support cosine-similarity comparisons.

### Step 5: Model Training

The training pipeline creates anchor, positive, and negative image samples.

Synthetic augmentations such as perspective transformations, rotations, cropping, horizontal flipping, color jitter, and random erasing introduce variation during training.

Triplet Margin Loss is used to optimize the embedding space.

### Step 6: Index Construction

The retrieval engine extracts embeddings for the images in the selected directory and adds them to a FAISS inner-product index.

Because the embeddings are normalized, inner-product scores correspond to cosine similarity.

### Step 7: Similarity Search

The query image is processed using the same embedding pipeline.

FAISS compares its embedding against the indexed vectors and returns the nearest matches.

### Step 8: Result Exploration

The command-line interface prints ranked image paths and similarity scores. The Streamlit dashboard provides a visual environment for exploring images and dynamically grouping related content.

---

## 📂 Project Architecture

The repository is organized into the following components:

```text
image-similarity-search/
│
├── dataset/
│   └── images/
│       └── sample.jpg
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   └── engine.py
│
├── temp_uploads/
│
├── app.py
├── train.py
├── search.py
├── dynamic_group_manager.py
│
├── requirements.txt
├── registry.json
├── embeddings.npy
│
├── model_weights.pth    # Generated after training
├── index.faiss          # Generated during indexing
└── paths.pkl            # Generated during indexing
```

*Some files and directories are generated at runtime and may not exist in a fresh clone. The structure above illustrates their intended roles.*

### File Descriptions

| File or Directory | Description |
|---|---|
| `app.py` | Main Streamlit dashboard for visual search and dynamic image grouping |
| `train.py` | Trains the embedding model using triplet-based metric learning |
| `search.py` | Command-line interface for indexing and retrieving similar images |
| `dynamic_group_manager.py` | Separate module for dynamic image-group management |
| `src/model.py` | Defines the ResNet-50 backbone and 256-dimensional projection head |
| `src/dataset.py` | Creates training triplets and applies image augmentations |
| `src/engine.py` | Extracts embeddings, builds the FAISS index, and performs retrieval |
| `dataset/images/` | Directory containing images used for training and retrieval |
| `temp_uploads/` | Stores uploaded query images during dashboard use |
| `registry.json` | Stores image registry and grouping metadata used by the dashboard |
| `embeddings.npy` | NumPy embedding artifact; its precise role depends on how it is generated and used |
| `model_weights.pth` | Model weights saved by the training script |
| `index.faiss` | Serialized FAISS vector index |
| `paths.pkl` | Serialized image paths corresponding to the FAISS index |
| `requirements.txt` | Python package dependencies |

---

## 💻 Prerequisites

Before running the project, ensure you have:

- Python 3.10 or later, with compatible package versions.
- Git.
- pip.
- At least 2 valid images in the dataset directory.
- Sufficient RAM for image processing and model execution.

A CUDA-compatible GPU is optional. CPU execution is supported, although training and large image collections may take longer.

---

## 📥 Installation and Setup

### Step 1: Clone the Repository

Open a terminal or PowerShell window and run:

```bash
git clone https://github.com/tanishka1308/image-similarity-search.git
```

Navigate to the project directory:

```bash
cd image-similarity-search
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you are using Command Prompt instead of PowerShell:

```bat
.venv\Scripts\activate.bat
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Install the packages listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current project dependencies include:

```text
torch
torchvision
faiss-cpu
pillow
numpy
scikit-learn
```

**Important:** The Streamlit dashboard also imports `streamlit` and `plotly`, which are not included in the current dependency list. Install them separately:

```bash
python -m pip install streamlit plotly
```

For a fresh setup, consider adding these two packages to `requirements.txt` so that all project dependencies can be installed with one command.

### Step 4: Verify the Installation

```bash
python -c "import torch, torchvision, faiss, numpy, PIL, streamlit, plotly; print('All required imports successful')"
```

If the command completes successfully, the main dependencies are available.

---

## 🖼️ Dataset Preparation

The project expects a directory of images for training and similarity retrieval.

Create the directory if it does not already exist:

```text
dataset/
└── images/
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.png
    ├── image4.jpeg
    └── image5.webp
```

Place your own image collection inside `dataset/images/`.

You can use product photographs, everyday objects, scenery, or another image collection relevant to your use case.

### Dataset Requirements

- Include at least two valid images for model training.
- Use supported image formats.
- Prefer readable, non-corrupted image files.
- Include a reasonably varied collection for more meaningful retrieval experiments.

**Note:** The current training pipeline creates positive samples by augmenting an image and chooses a different image as the negative sample. It does not use manually annotated semantic labels. Consequently, meaningful visual similarity is not guaranteed for every dataset.

---

## 🧠 Training the Model

Once your dataset is ready, execute:

```bash
python train.py
```

The training script:

1. Detects whether CUDA is available.
2. Loads the dataset from `dataset/images/`.
3. Creates batches of anchor, positive, and negative images.
4. Initializes the ResNet-50 embedding model.
5. Optimizes the network using Triplet Margin Loss.
6. Saves the trained model weights.

### Current Training Configuration

| Parameter | Value |
|---|---|
| Embedding dimension | 256 |
| Batch size | 16 |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Triplet margin | 0.2 |
| Device | CUDA when available; otherwise CPU |

The trained weights are saved to:

```text
model_weights.pth
```

The training script prints the average loss for each epoch.

**Important:** Five epochs are the current default configuration, not a guarantee of optimal model performance. Actual training quality depends on the dataset, initialization, and training procedure.

---

## ▶️ Running the Application

The project has a Streamlit dashboard and a separate command-line search workflow.

### Option 1: Launch the Interactive Dashboard

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will display a local address in your terminal. Open that address in your browser.

The default local address is usually:

```text
http://localhost:8501
```

### Dashboard Workflow

**1. Choose a visual theme**

Select Cyber Dark, Midnight Purple, or Clean Light Glass from the sidebar.

**2. Configure similarity**

Adjust the cosine-similarity cutoff using the sidebar slider.

The current default is `0.55`, with a configurable range from `0.30` to `0.95`.

**3. Select an image**

Choose an existing image from the registry or upload a new image.

**4. Run the search**

Click the dynamic search and grouping button.

**5. Explore the results**

Review the images, inspect their groups, and navigate the available dashboard tabs:

- Search & Grouping
- Active Clusters
- Origin Pool Inventory

**6. Manage the collection**

Use the available grouping controls to explore the image organization. The reset control restores grouped images to the origin pool and clears the grouping registry.

> **Model loading:** The dashboard loads `model_weights.pth` if it exists. If the file is absent, it initializes the model with the default pretrained ResNet-50 backbone and a newly initialized projection head. For consistency, train the model before using it for retrieval.

---

## 🔎 Using the Image Search CLI

The command-line interface supports direct similarity searches over a directory of images.

### Basic Command

```bash
python search.py --query dataset/images/image1.jpg --dir dataset/images --index --top_k 5
```

### Available Arguments

| Argument | Required? | Description |
|---|---|---|
| `--query` | Yes | Path to the query image |
| `--dir` | No | Directory containing candidate images; defaults to `dataset` |
| `--index` | No | Rebuild the FAISS index before searching |
| `--top_k` | No | Number of nearest matches to retrieve; defaults to 5 |

### Example: Find the Top 10 Similar Images

```bash
python search.py --query dataset/images/image1.jpg --dir dataset/images --index --top_k 10
```

This command rebuilds the index from the selected directory and retrieves up to ten matches.

### Reuse an Existing Index

After building the index once, you can run:

```bash
python search.py --query dataset/images/image2.jpg --dir dataset/images --top_k 5
```

If `index.faiss` already exists, the script loads the saved index instead of rebuilding it.

The corresponding `paths.pkl` file must also exist and match the index.

### Example Output

The output follows this general format:

```text
Loaded trained model_weights.pth

Querying: dataset/images/image1.jpg

--- Top Matches ---
1. dataset/images/image3.jpg | Similarity: 0.9234
2. dataset/images/image5.png | Similarity: 0.8612
3. dataset/images/image2.jpg | Similarity: 0.7845
```

*The scores and filenames above are illustrative examples, not measured results from the repository.*

---

## 📊 Similarity Score Explained

The retrieval engine uses cosine similarity to compare normalized image embeddings.

For vectors \(A\) and \(B\), cosine similarity is:

\[
\operatorname{cosine}(A,B)=
\frac{A \cdot B}{\|A\|\|B\|}
\]

Because the model normalizes the embeddings to unit length, the inner product between two embeddings corresponds to their cosine similarity.

### Interpreting Scores

| Score | General Interpretation |
|---|---|
| Closer to 1 | Embeddings point in a similar direction |
| Around 0 | Embeddings have little directional similarity |
| Closer to -1 | Embeddings point in opposing directions |

These are mathematical interpretations, not fixed semantic categories.

The model's scores are not calibrated probabilities. A score of `0.90` does not mean there is a 90% probability that two images depict the same object.

### Similarity Threshold

The dashboard uses a configurable threshold to support its grouping workflow.

A higher threshold generally imposes a stricter similarity requirement, while a lower threshold allows more candidate images to qualify.

The usefulness of a particular threshold depends on the dataset and the quality of the learned embeddings.

---

## 📁 Configuration and Generated Files

### `model_weights.pth`

Contains the trained model parameters produced by `train.py`.

Train the model again whenever you want to experiment with a new training configuration.

### `index.faiss`

Contains the serialized vector index used by the CLI retrieval engine.

Rebuild it when the indexed image collection changes or when you want to index embeddings from updated model weights.

### `paths.pkl`

Contains the image paths associated with the index entries.

Keep it synchronized with `index.faiss`. Rebuilding the index also regenerates the corresponding path file.

### `registry.json`

Stores image paths and grouping information used by the dashboard.

If you manually move or rename images, the registry may need to be rebuilt or reset so that its paths remain valid.

### `embeddings.npy`

A NumPy array artifact present in the repository. Check the code that generates or consumes this file before deleting or replacing it.

---

## 🛠️ Troubleshooting

### 1. `ModuleNotFoundError: No module named 'streamlit'`

Install Streamlit:

```bash
python -m pip install streamlit
```

### 2. `ModuleNotFoundError: No module named 'plotly'`

Install Plotly:

```bash
python -m pip install plotly
```

### 3. Dataset Directory Not Found

Ensure this directory exists:

```text
dataset/images/
```

The current training script expects this directory to be available.

### 4. Dataset Must Contain at Least Two Images

The training dataset requires at least two valid image files.

Add more images and ensure they use supported formats.

### 5. Model Weights Not Found

If `model_weights.pth` does not exist, the search CLI falls back to the model's default pretrained ResNet-50 backbone and its newly initialized projection head.

To create trained weights, run:

```bash
python train.py
```

### 6. FAISS Index Not Found

Build the index before searching:

```bash
python search.py --query dataset/images/image1.jpg --dir dataset/images --index --top_k 5
```

### 7. FAISS Index and Image Paths Do Not Match

Rebuild the index after changing the image collection:

```bash
python search.py --query dataset/images/image1.jpg --dir dataset/images --index --top_k 5
```

### 8. CUDA Is Not Available

The project can run on a CPU.

If you want GPU acceleration, check your GPU compatibility and install a PyTorch build appropriate for your CUDA environment.

### 9. Search Results Are Not Semantically Meaningful

Check the following:

- Whether the model was trained successfully.
- Whether the dataset contains sufficiently varied images.
- Whether the index was rebuilt after retraining.
- Whether the similarity threshold is appropriate.
- Whether the triplet-learning strategy is suitable for the dataset.

---

## 🌍 Applications and Use Cases

The underlying techniques can be applied to several practical problems:

- **E-commerce:** Find visually similar products.
- **Digital asset management:** Organize large image libraries.
- **Duplicate discovery:** Identify images with similar visual representations.
- **Photography:** Explore related photographs.
- **Content organization:** Group images with similar visual characteristics.
- **Research:** Retrieve related images from a reference collection.

These are potential applications of the underlying approach; the repository is not a validated production system for each use case.

---

## 🔮 Future Improvements

Possible extensions include:

- [ ] Add a browser-based interface for previewing ranked CLI results.
- [ ] Introduce a more representative, labeled training dataset.
- [ ] Evaluate retrieval quality using Precision@K, Recall@K, and mean Average Precision.
- [ ] Compare the current model with pretrained CLIP or other vision embeddings.
- [ ] Add support for larger datasets using approximate nearest-neighbor indexing.
- [ ] Add image metadata and filtering options.
- [ ] Improve index management when images are added, removed, or renamed.
- [ ] Add automated tests for preprocessing, training, indexing, and retrieval.
- [ ] Provide a hosted demo and deployment instructions.
- [ ] Add dataset and model configuration through a dedicated configuration file.

---

## 👩‍💻 Author

**Tanishka Sharma**

Computer Science & Engineering — Artificial Intelligence and Machine Learning

GitHub: [@tanishka1308](https://github.com/tanishka1308)

### Feedback and Contributions

Suggestions, bug reports, and improvements are welcome.

If you encounter an issue, please include the error message, relevant environment details, and steps to reproduce it.

---

## 📄 License

No license is currently documented in this README.

Before distributing or reusing the project, review the licensing terms of the source code, pretrained model weights, and dataset. Add an appropriate `LICENSE` file if you intend to publish the project under an open-source license.
