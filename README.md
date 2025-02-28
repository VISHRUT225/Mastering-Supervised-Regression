# 📊 Mastering Supervised Regression

🚀 **End-to-End Regression Modeling Project**

This repository contains an end-to-end supervised regression pipeline, covering data preprocessing, feature engineering, model training, evaluation, and deployment. The goal is to build an optimized regression model and compare various algorithms.

---

## 📂 Folder Structure

```
📦 Mastering-Supervised-Regression
├── 📁 code                # Jupyter Notebook and scripts
│   ├── end-to-end-regression-modeling.ipynb
├── 📁 report              # Documentation and reports
│   ├── Mastering Supervised Regression.pdf
├── 📁 model_and_results   # Saved models and evaluation results
│   ├── best_model.pkl
│   ├── RMSE_comparison.png
│   ├── R2_comparison.png
├── README.md              # Project overview and guide
```

---

## 📜 Project Overview

This project demonstrates the process of building and fine-tuning regression models. The dataset is processed through various stages, including: 

✅ **Data Preprocessing** (Handling missing values, outliers, and scaling)  
✅ **Feature Engineering** (Creating new features, encoding categorical variables)  
✅ **Model Training & Hyperparameter Tuning** (Ridge, Lasso, SVR, Decision Trees, Random Forest, KNN)  
✅ **Performance Evaluation** (Metrics like MSE, RMSE, R², visualization of results)  
✅ **Deployment** (Saving the model and making predictions)  

---

## 📊 Dataset & Preprocessing

- Data Cleaning: Handling missing values and outliers.  
- Feature Engineering: Encoding categorical variables, feature selection.  
- Train-Test Split: Using an 80/20 split.  

🔹 **Example Visualization:**  
![Dataset Visualization](model_and_results/data_visualization.png)

---

## 🤖 Model Training & Evaluation

### 🏆 Trained Models:
- **Linear Regression**
- **Ridge Regression (L2 Regularization)**
- **Lasso Regression (L1 Regularization)**
- **Polynomial Regression**
- **Support Vector Regression (SVR)**
- **Decision Tree Regression**
- **Random Forest Regression**
- **K-Nearest Neighbors (KNN)**

### 📈 Model Performance (Evaluation Metrics):
| Model          | MSE  | RMSE | MAE  | R²  |
|---------------|------|------|------|-----|
| Ridge         | 0.10 | 0.31 | 0.20 | 0.91 |
| Lasso         | 0.10 | 0.31 | 0.20 | 0.91 |
| Polynomial    | 0.15 | 0.39 | 0.29 | 0.86 |
| SVR           | 0.12 | 0.35 | 0.19 | 0.89 |
| Decision Tree | 0.15 | 0.39 | 0.28 | 0.86 |
| Random Forest | 0.08 | 0.28 | 0.19 | 0.93 |
| KNN           | 0.15 | 0.39 | 0.28 | 0.86 |

---

## 🎯 Key Takeaways

✔ **Random Forest** achieved the best performance with the lowest error and highest R².  
✔ **Ridge & Lasso** performed well, with Ridge handling multicollinearity effectively.  
✔ **Polynomial Regression, Decision Tree, and KNN** had higher errors, indicating overfitting.  
✔ **Feature scaling significantly impacted SVR & KNN performance.**  

---

## 🚀 Model Deployment

- The best model is saved as `best_model.pkl` using `joblib`.
- Example Prediction Function:

```python
from joblib import load

# Load the model
model = load("model_and_results/best_model.pkl")

# Example input (must match training feature format)
example_input = [[3, 1200, 2, 1]]

# Make prediction
predicted_value = model.predict(example_input)
print(f"Predicted Value: {predicted_value}")
```

---

## 📌 Future Improvements

🔹 Feature Selection using advanced techniques (e.g., SHAP values)  
🔹 Experimenting with deep learning regression models  
🔹 Hyperparameter tuning using Bayesian Optimization  

---

## 📜 References

- Scikit-learn Documentation  
- Kaggle Datasets  
- Matplotlib & Seaborn for Visualization  

💡 **Contributions & Feedback Welcome!**  

---

📢 _If you find this project useful, consider giving it a ⭐ on GitHub!_ 🚀
