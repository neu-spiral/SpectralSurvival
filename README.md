# Spectral Survival Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **"Spectral Survival Analysis"** published in the Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25).

Survival analysis is widely deployed in a diverse set of fields, including healthcare, business, ecology, etc. The Cox Proportional Hazard (CoxPH) model is a semi-parametric model often encountered in the literature. Despite its popularity, wide deployment, and numerous variants, scaling CoxPH to large datasets and deep architectures poses a challenge, especially in the high-dimensional regime. We identify a fundamental connection between rank regression and the CoxPH model: this allows us to adapt and extend the so-called spectral method for rank regression to survival analysis. Our approach is versatile, naturally generalizing to several CoxPH variants, including deep models. We empirically verify our method's scalability on multiple real-world high-dimensional datasets; our method outperforms legacy methods w.r.t. predictive performance and efficiency.

## 🚀 Key Features

- **Novel Spectral Method**: Leverages the connection between rank regression and Cox Proportional Hazard models
- **Scalability**: Efficiently handles large datasets and high-dimensional data
- **Multiple Architectures**: Supports both MLP and convolutional neural networks (1D ResNet, 3D ConvNet)
- **Comprehensive Baselines**: Includes implementations of DeepSurv, DeepHit, MTLR, CoxCC, FastCPH, and CoxTime
- **Multiple Datasets**: Supports various survival analysis datasets such as LUNG1, DLBCL, MovieLens, etc.

## 🏗️ Architecture

The repository implements:

1. **Spectral Algorithm** (`methods/survmodel_pytorch.py`): Core spectral survival analysis implementation
2. **Baseline Methods** (`methods/baseline.py`): Standard survival analysis methods for comparison
3. **Neural Network Models** (`methods/models.py`): MLP, ResNet1D, and 3D ConvNet implementations
4. **Evaluation Metrics** (`methods/metrics.py`): Concordance index, integrated Brier score, AUC, and survival MSE

## 📊 Supported Datasets

- **Medical Datasets**: METABRIC, SUPPORT, WHAS, BraTS2020
- **Synthetic Datasets**: Simulated survival data with configurable parameters
- **MovieLens**: Recommendation data adapted for survival analysis
- **High-dimensional Datasets**: LUNG1, DLBCL, VDV

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/KDD_Spectral_Survival_Analysis.git
cd KDD_Spectral_Survival_Analysis
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn scipy
pip install pycox torchtuples scikit-survival lifelines
pip install deepsurvk lassonet wandb
```

## 🚀 Quick Start

### Running the Spectral Algorithm

```bash
# Run spectral algorithm on DLBCL dataset
python main.py --dataset DLBCL --algorithm spectral --learning_rate 0.01 --epochs 200

# Run with ResNet architecture
python main.py --dataset DLBCL --algorithm spectral --ResNet --depth 3 --epochs 100
```

### Running Baseline Methods

```bash
# Run DeepSurv baseline
python main.py --dataset DLBCL --algorithm deepsurv --learning_rate 0.01

# Run all baseline methods
python main.py --dataset DLBCL --algorithm baseline --epochs 150
```

### MovieLens Experiments

```bash
# Run spectral method on MovieLens data
python main_movie.py --dataset MovieLens100k --algorithm spectral --epochs 50

# Run DeepSurv on MovieLens
python main_movie.py --dataset MovieLens100k --algorithm deepsurv --batch_size_coef 0.01
```

## 📋 Command Line Arguments

### Main Arguments
- `--dataset`: Dataset name (metabric, support, whas, DLBCL, MovieLens100k, etc.)
- `--algorithm`: Algorithm to run (spectral, deepsurv, deephit, mtlr, coxcc, fastcph, coxtime, baseline)
- `--learning_rate`: Learning rate (default: 0.01)
- `--epochs`: Number of training epochs (default: 200)
- `--depth`: Network depth (default: 3)
- `--dropout`: Dropout rate (default: 0.5)
- `--batch_size_coef`: Batch size coefficient (default: 1.0)

### Spectral-Specific Arguments
- `--rtol`: Relative tolerance for convergence (default: 1e-4)
- `--n_iter`: Maximum iterations for inner loop (default: 100)

### Architecture Arguments
- `--ResNet`: Use ResNet architecture
- `--dims`: Hidden layer dimensions

## 📈 Evaluation Metrics

The implementation provides comprehensive evaluation using:

- **Concordance Index (CI)**: Measures ranking quality
- **Integrated Brier Score (IBS)**: Evaluates prediction accuracy over time
- **Time-dependent AUC**: Area under ROC curve at specific time points
- **RMSE**: Root of MSE for survival function estimates

## 🔬 Experimental Results

The spectral method demonstrates:
- Superior scalability on high-dimensional datasets
- Competitive or better performance compared to baseline methods
- Efficient training with reduced computational overhead
- Robust performance across different dataset types

## 📁 Project Structure

```
├── main.py                    # Main training script
├── main_movie.py             # MovieLens-specific experiments
├── methods/
│   ├── survmodel_pytorch.py  # Spectral survival analysis implementation
│   ├── baseline.py          # Baseline survival methods
│   ├── models.py            # Neural network architectures
│   ├── metrics.py           # Evaluation metrics
│   ├── preprocessing.py     # Data preprocessing utilities
│   ├── utils.py             # Utility functions
│   └── plot.py              # Visualization tools
└── README.md                # This file
```

## 🎯 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{10.1145/3711896.3737134,
author = {Shi, Chengzhi and Ioannidis, Stratis},
title = {Spectral Survival Analysis},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3737134},
doi = {10.1145/3711896.3737134},
abstract = {Survival analysis is widely deployed in a diverse set of fields, including healthcare, business, ecology, etc. The Cox Proportional Hazard (CoxPH) model is a semi-parametric model often encountered in the literature. Despite its popularity, wide deployment, and numerous variants, scaling CoxPH to large datasets and deep architectures poses a challenge, especially in the high-dimensional regime. We identify a fundamental connection between rank regression and the CoxPH model: this allows us to adapt and extend the so-called spectral method for rank regression to survival analysis. Our approach is versatile, naturally generalizing to several CoxPH variants, including deep models. We empirically verify our method's scalability on multiple real-world high-dimensional datasets; our method outperforms legacy methods w.r.t. predictive performance and efficiency.},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {2538–2549},
numpages = {12},
keywords = {neural networks, spectral methods, survival analysis},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Chengzhi Shi** 
- **Stratis Ioannidis**

## 📞 Contact

For questions and support, please open an issue in this repository or contact the authors at shi.cheng@northeastern.edu.

## 🙏 Acknowledgments

- The authors gratefully acknowledge support from the National
Science Foundation (grants 2112471 and 1750539)
- Built upon PyTorch, scikit-survival, and pycox libraries


