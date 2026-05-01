# 🏃 Walk vs Run Classification using Machine Learning

## 📌 Overview

This project focuses on building a machine learning model to classify human physical activity as **Walking (0)** or **Running (1)** using wearable sensor data. The dataset contains acceleration and gyroscope readings collected from motion sensors.

The goal is to design a robust and interpretable ML pipeline that can accurately distinguish between different movement patterns.

---

## 🎯 Objectives

* Perform **Exploratory Data Analysis (EDA)** to understand sensor behavior
* Apply **feature engineering** to enhance motion representation
* Train and compare multiple machine learning models
* Evaluate performance using appropriate metrics
* Interpret model decisions using feature importance and SHAP
* Build a scalable pipeline for real-world activity recognition

---

## 📊 Dataset Description

The dataset includes:

* **Acceleration Data**: `acceleration_x`, `acceleration_y`, `acceleration_z`
* **Gyroscope Data**: `gyro_x`, `gyro_y`, `gyro_z`
* **Metadata**: `date`, `time`, `username`, `wrist`
* **Target Variable**:

  * `activity` → 0 (Walking), 1 (Running)

---

## 🔍 Exploratory Data Analysis (EDA)

* Checked data structure, types, and missing values
* Visualized **feature distributions** using histograms
* Analyzed **target distribution** (balanced dataset ~50/50)
* Generated **correlation heatmap** to understand feature relationships
* Identified potential outliers using boxplots

### 💡 Key Insights

* Dataset is **balanced**, eliminating the need for resampling
* `acceleration_y` shows strong correlation with activity
* Gyroscope features contribute less individually
* No severe multicollinearity observed

---

## 🔥 Feature Engineering

To better capture motion patterns:

* Created **acceleration magnitude**:

  ```
  sqrt(x² + y² + z²)
  ```
* Created **gyroscope magnitude**

### 💡 Insight

Magnitude features represent overall movement intensity and significantly improve model performance.

---

## ⚙️ Preprocessing

* Removed non-numeric columns (`date`, `time`, `username`)
* Split dataset using **stratified sampling**
* Applied **StandardScaler** for normalization

---

## 🤖 Models Implemented

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* Neural Network (MLP)

---

## 🏆 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Neural Network      | ~99.27%  |
| Random Forest       | ~99.18%  |
| KNN                 | ~99.13%  |
| SVM                 | ~99.10%  |
| Gradient Boosting   | ~98.40%  |
| Logistic Regression | ~86.39%  |

### ✅ Best Model: Random Forest

* Handles non-linear relationships effectively
* Robust to noise and outliers
* Provides feature importance for interpretability

---

## 🔧 Hyperparameter Tuning

Used **GridSearchCV** to optimize Random Forest parameters:

* `n_estimators`
* `max_depth`

---

## 📈 Model Evaluation

* Accuracy: ~99.21%
* Precision, Recall, F1-score: ~0.99 for both classes
* Confusion Matrix shows minimal misclassification

### 🔁 Cross-Validation

* 5-fold CV accuracy: ~99.13%
* Confirms strong generalization and no overfitting

---

## 🔍 Model Interpretability

### Feature Importance

* `acceleration_y` is the most important feature
* Magnitude features enhance predictive power

### SHAP Analysis

* Provided deeper insight into feature contributions
* Confirmed that vertical motion and intensity drive predictions

---

## 🧪 Outlier Analysis

* IQR method removed ~21.7% of data
* Model performance slightly decreased after removal

### 💡 Insight

Outliers represented **real human motion variations**, not noise
➡️ Retaining them improved model performance

---

## 🧪 Sample Prediction Validation

* Tested on 25 random unseen samples
* Achieved 100% accuracy on sample

⚠️ Note: Small sample accuracy is not reliable; overall test and CV scores are better indicators.

---

## 🚀 Key Learnings

* Feature engineering is critical for sensor data
* Tree-based models outperform linear models in non-linear problems
* Outliers in real-world data may contain valuable information
* Proper evaluation requires both test data and cross-validation

---

## 🔮 Future Improvements

* Implement **LSTM / RNN** for time-series modeling
* Deploy model using **Flask / FastAPI**
* Real-time activity tracking using wearable devices
* Advanced feature engineering (rolling windows, frequency features)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* SHAP

---

## 📌 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for activity recognition. With strong performance (~99% accuracy), robust evaluation, and interpretability, the model is well-suited for real-world applications such as fitness tracking and wearable analytics.

---

## 👤 Author

**Vrund Patel**
Aspiring AI Engineer | Machine Learning Enthusiast
