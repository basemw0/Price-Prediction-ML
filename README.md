# Data Prediction Project

## Introduction
This project explores different regression models to predict a target variable (Y) based on a given dataset. The models used include:

- **Support Vector Regression (SVR)**
- **Polynomial Regression**
- **Random Forest Regression**
- **XGBoost Regression**

Each model was trained and evaluated using different preprocessing techniques and hyperparameter tuning to optimize performance.

---

## Data Preprocessing
- **Handling Missing Values**:
  - `X9` (categorical) was filled with the mode.
  - `X2` (numerical) was filled with the mean.
- **Feature Selection**:
  - `X1` (product ID) was dropped due to lack of relevance.
  - `X2` and `X8` were dropped due to weak correlation with `Y`.
- **One-Hot Encoding**:
  - `X5`, `X7`, and `X11` were one-hot encoded.
  - Only `X7_0`, `X7_4`, `X11_0`, `X11_1`, and `X11_3` were kept based on correlation.
- **Label Encoding**:
  - `X3`, `X9`, and `X10` were label-encoded but later dropped due to weak correlation.
- **Standardization and Normalization**:
  - `X6` was standardized for better model performance.

---

## Models & Performance

### 1️⃣ Support Vector Regression (SVR) - **Best Performing Model**
**Score:** *0.370*

- Used **Grid Search with 5-fold Cross-Validation** for hyperparameter tuning.
- Key parameters: `C = 50`, `epsilon = 0.005`, `gamma = 0.1`, `kernel = rbf`.
- Standardization applied to `X6`.

### 2️⃣ Polynomial Regression
**Score:** *0.400*

- Polynomial features with degree **3** were selected.
- Normalization applied to `X6`.

### 3️⃣ Random Forest Regression
**Score:** *0.385*

- **Grid Search Optimization** resulted in:
  - `max_depth = 10`
  - `n_estimators = 200`
  - `min_samples_split = 10`
- Standard preprocessing applied.

### 4️⃣ XGBoost Regression
**Score:** *0.481*

- Used the same preprocessing steps as **Polynomial Regression**.
- Key parameters: `n_estimators = 200`, `learning_rate = 0.3`, `max_depth = 6`.

---

## Conclusion
| Model | Score |
|----------------------|--------|
| **Support Vector Regression (SVR)** | **0.370 (Best Performance)** |
| Polynomial Regression | 0.400 |
| Random Forest Regression | 0.385 |
| XGBoost Regression | 0.481 |

- **SVR provided the best overall results**, balancing accuracy and model reliability.
- Polynomial Regression also showed strong results.
- Feature selection and hyperparameter tuning played a crucial role in performance.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/data-prediction-project.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```

---

## 📌 Future Improvements
- **Hyperparameter Optimization**: Fine-tuning with Bayesian Optimization.
- **More Models**: Exploring Neural Networks and LightGBM.

---

### ✨ Author
[Basem Walid](https://github.com/basemw0)


