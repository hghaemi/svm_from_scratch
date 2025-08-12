# Support Vector Machine (SVM) Implementation from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, educational implementation of Support Vector Machine (SVM) classifier built from scratch using only NumPy for core computations. This project demonstrates the mathematical foundations and optimization process behind one of the most powerful machine learning algorithms.

## 📚 Overview

This implementation focuses on **binary classification** using the **linear SVM** with **soft margin** approach. The project is designed for educational purposes, providing clear insights into:

- **Gradient Descent Optimization**: Custom implementation of the SVM optimization problem
- **Margin Maximization**: Understanding how SVMs find the optimal separating hyperplane
- **Mathematical Foundation**: Direct translation of SVM theory into code

## 🎯 Key Features

- **Pure NumPy Implementation**: No external ML libraries for core algorithm
- **Soft Margin SVM**: Handles non-linearly separable data
- **Visualization Tools**: Interactive plots showing decision boundary and support vectors
- **Performance Metrics**: Comprehensive evaluation using accuracy, precision, recall, and F1-score
- **Educational Focus**: Well-commented code explaining each mathematical step

## 🔧 Requirements

```
numpy>=1.21.0
scikit-learn>=1.0.0    # Only for data generation and metrics
matplotlib>=3.5.0      # For visualization
jupyter>=1.0.0         # For notebook interface
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hghaemi/svm_from_scratch.git
cd svm_from_scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook svm.ipynb
```

### Basic Usage

```python
from svm import SVM
import numpy as np
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert to {-1, 1} labels

# Initialize and train SVM
clf = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X_new)
```

## 🧮 Mathematical Foundation

The SVM optimization problem is formulated as:

**Minimize:** `½||w||² + C∑ξᵢ`

**Subject to:** `yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ`

Where:
- `w`: weight vector defining the hyperplane
- `b`: bias term
- `C`: regularization parameter (λ in our implementation)
- `ξᵢ`: slack variables for soft margin

### Gradient Descent Update Rules

For each training sample `(xᵢ, yᵢ)`:

**If** `yᵢ(wᵀxᵢ + b) ≥ 1` (correctly classified):
- `w ← w - η(2λw)`

**Else** (misclassified or within margin):
- `w ← w - η(2λw - yᵢxᵢ)`
- `b ← b - η(-yᵢ)`

## 📊 Performance Analysis

The implementation includes comprehensive evaluation metrics:

- **Accuracy Score**: Overall classification performance
- **Confusion Matrix**: Detailed breakdown of predictions
- **Classification Report**: Precision, recall, and F1-score
- **Decision Boundary Visualization**: Visual representation of the learned hyperplane

### Sample Results

```
Accuracy: 0.95
              precision    recall  f1-score   support
          -1       0.94      0.96      0.95         8
           1       0.96      0.94      0.95         9
    accuracy                           0.95        17
   macro avg       0.95      0.95      0.95        17
weighted avg       0.95      0.95      0.95        17
```

## 🎨 Visualization Features

The project includes sophisticated visualization capabilities:

1. **Data Distribution**: Scatter plot of training data with class labels
2. **Decision Boundary**: The optimal separating hyperplane
3. **Margin Boundaries**: Upper and lower margin boundaries (±1 from decision boundary)
4. **Support Vectors**: Highlighted data points that define the margin

## 📈 Hyperparameter Tuning

Key parameters to experiment with:

- **`learning_rate`** (default: 0.001): Controls optimization step size
  - Higher values: Faster convergence, risk of overshooting
  - Lower values: More stable, slower convergence

- **`lambda_param`** (default: 0.01): Regularization strength
  - Higher values: More regularization, wider margin
  - Lower values: Less regularization, narrower margin

- **`n_iters`** (default: 1000): Number of training iterations
  - More iterations: Better convergence, longer training time

## 🔬 Educational Applications

This implementation is ideal for:

- **Machine Learning Courses**: Understanding SVM fundamentals
- **Academic Research**: Baseline implementation for algorithm modifications
- **Interview Preparation**: Demonstrating ML algorithm implementation skills
- **Algorithm Analysis**: Studying optimization behavior and convergence

## 🏗️ Project Structure

```
svm-from-scratch/
│
├── svm.ipynb              # Main implementation notebook
├── requirements.txt       # Package dependencies
├── README.md             # Project documentation
└── LICENSE               # MIT license
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- **Kernel Methods**: Implement RBF, polynomial kernels
- **Multi-class Extension**: One-vs-Rest or One-vs-One approaches  
- **Performance Optimization**: Vectorized operations, early stopping
- **Advanced Visualizations**: 3D plots, learning curves

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by classic SVM literature and educational resources
- Built with the Python scientific computing ecosystem
- Designed for academic and educational use

---

**Note**: This implementation prioritizes educational clarity over production performance. For real-world applications, consider using optimized libraries like scikit-learn or libsvm.