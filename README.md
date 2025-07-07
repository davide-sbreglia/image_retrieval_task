# Top-k Image Retrieval Pipeline

This repository contains our solution for the Image Retrieval Competition in the Introduction to Machine Learning course at the University of Trento (2025). The goal is to retrieve the top-k most similar gallery images for each query image, matching real celebrity photos to synthetic renders.

## Project Structure

```
ML Project/
├── Dataset/           # Competition dataset (train, test/query, test/gallery)
├── MLOps_project/
│   ├── project/
│   │   └── src/       # Main code: training, retrieval, ensemble, etc.
│   ├── report/        # Report sections (LaTeX)
│   └── ...            
```

## Main Features

- **Backbones:** ResNet-50, EfficientNet-B0, ViT-B/16 (ImageNet pre-trained)
- **Training:** Fine-tuning only the classifier head; backbone frozen
- **Retrieval:** Embeddings projected to 512-D, L2-normalized, compared via cosine similarity
- **Indexing:** Fast nearest-neighbor search with FAISS
- **Automated pipeline:** Full workflow via `run_pipeline.sh`

## How to Run

1. **Install dependencies**  
   (Recommended: Python 3.8+, PyTorch, torchvision, faiss, etc.)
   ```
   pip install -r requirements.txt
   ```

2. **Prepare the dataset**  
   Place the competition dataset in the `Dataset/` folder, following the structure:
   ```
   Dataset/
   ├── train/
   └── test/
       ├── query/
       └── gallery/
   ```

3. **Run the full pipeline**
   ```
   cd MLOps_project/project
   bash run_pipeline.sh <path_to_train> <path_to_test>
   ```
   This script splits the data, benchmarks models, trains, indexes, retrieves, and generates the submission file.

4. **Submission**  
   The output `submission.json` can be uploaded to the competition portal.

## Key Scripts

- `src/main.py` — Unified CLI for splitting, training, indexing, and retrieval
- `src/train_ft.py` — Fine-tuning the classifier head
- `src/ensemble_retrieve.py` — Ensemble retrieval (optional)
- `src/wandb_benchmark.py` — Model benchmarking and selection

## Evaluation

- **Competition metric:** Weighted Top-k accuracy (Top-1, Top-5, Top-10)
- **Validation:** Automated model selection on a held-out split
- **Qualitative analysis:** Visualizations of retrieval results

## Report

A detailed technical report is provided in the `MLOps_project/` folder, including methodology, experiments, and results.

## Authors

Project developed by Camilla Bonomo, Paolo Fabbri, Davide Sbreglia, University of Trento (2025).

