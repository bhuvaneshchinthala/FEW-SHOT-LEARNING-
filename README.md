# Few-Shot Learning for Novel Object Detection in Autonomous Driving

A deep learning project implementing few-shot learning techniques to detect novel objects in autonomous driving scenarios.

## 🎯 Overview

This project implements a few-shot object detection model that can learn to recognize new object categories with only a few training examples (K-shot learning). It uses a two-stage training approach:

1. **Stage 1**: Base model training on common autonomous driving objects
2. **Stage 2**: Few-shot fine-tuning to learn novel object categories

## 🏗️ Model Architecture

- **Backbone**: CSPNet (Cross Stage Partial Network)
- **Encoder**: Dilated Convolution blocks with rates [2, 4, 6, 8]
- **Detection Head**: Decoupled Head with separate classification and regression branches
- **Classifier**: Cosine Similarity Classifier for few-shot learning

## 📊 Datasets

### Base Classes (BDD100K)
Training on 9 common autonomous driving categories:
- Car, Truck, Bus, Person, Bike, Motor, Traffic Light, Traffic Sign, Rider

### Novel Classes (Pascal VOC)
Few-shot learning on 6 novel categories:
- Dog, Cat, Cow, Horse, Sheep, Bird

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 384 × 640 |
| K-Shot | 5 |
| Stage 1 Epochs | 30 |
| Stage 2 Epochs | 20 |
| Stage 1 Learning Rate | 1e-4 |
| Stage 2 Learning Rate | 1e-5 |
| Classification Loss | Focal Loss (α=0.25, γ=2.0) |
| Regression Loss | CIoU Loss |

## 📈 Training Pipeline

### Stage 1: Base Model Training
- Dataset: BDD100K (8000 train, 2000 val images)
- Optimizer: AdamW with weight decay 1e-4
- Scheduler: Cosine Annealing

### Stage 2: Few-Shot Fine-tuning
- Dataset: Balanced few-shot dataset with K samples per class
- Optimizer: Adam
- Backbone frozen, only detection head is fine-tuned
- Uses Cosine Classifier for better generalization

## 📉 Loss Functions

- **Classification**: Focal Loss for handling class imbalance
- **Objectness**: Focal Loss
- **Bounding Box**: CIoU (Complete IoU) Loss

## 🔧 Requirements

```bash
pip install ultralytics seaborn scikit-learn torch torchvision kagglehub
```

## � Dataset Download

The datasets are too large to store on GitHub. Use the following code to download them:

```python
import kagglehub

# Download BDD100K dataset
bdd_path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
print("BDD100K dataset path:", bdd_path)

# Download Pascal VOC 2012 dataset
voc_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")
print("Pascal VOC dataset path:", voc_path)
```

**Note**: You need to have a Kaggle account and API credentials configured. See [Kaggle API documentation](https://www.kaggle.com/docs/api) for setup instructions.

## �🚀 Usage

Run the Jupyter notebook `few-shot-learning (1).ipynb` to:

1. Download and prepare datasets
2. Train the base model
3. Perform few-shot fine-tuning
4. Evaluate and visualize results

## 📊 Evaluation Metrics

The model is evaluated using:
- Per-class accuracy
- Confusion matrix
- Detection confidence analysis

## 📁 Project Structure

```
├── few-shot-learning (1).ipynb   # Main training notebook
├── README.md                      # This file
└── outputs/                       # Model checkpoints and results
    ├── base_model.pth
    ├── fewshot_model.pth
    └── confusion_matrix.png
```

## 🎮 Hardware Requirements

- GPU recommended (Tesla T4 or similar)
- ~16GB GPU memory for training
- Training time: ~2-3 hours on T4 GPU

## 📚 References

- BDD100K Dataset
- Pascal VOC 2012 Dataset
- CSPNet Architecture
- Focal Loss for Dense Object Detection
- Few-Shot Object Detection Methods

## 📝 License

This project is for educational purposes as part of SEM 6 project work.

## 👤 Author

Bhuvanesh Chinthala
