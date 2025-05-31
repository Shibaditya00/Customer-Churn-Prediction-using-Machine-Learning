# Customer Churn Prediction Using Machine Learning

This project aims to build a predictive model to identify customers who are likely to stop using a telecom company's services. By leveraging machine learning techniques, we can analyze customer behavior and predict churn with high accuracy. This helps businesses to proactively retain customers and reduce churn rates.

🔗 Project Repository: <a href="https://github.com/Shibaditya00/Customer-Churn-Prediction-using-Machine-Learning">Customer Churn Prediction using Machine Learning</a>

---

Dataset: <a href="https://github.com/Shibaditya00/Customer-Churn-Prediction-using-Machine-Learning/blob/main/WA_Fn-UseC_-Telco-Customer-Churn.csv">Dataset Link</a>

--- 

## 🚀 Workflow

The project follows a structured data science workflow:

1. 📥 Data Collection

   * Dataset used: Telco Customer Churn Dataset from Kaggle or IBM Sample Data.
   * It includes customer demographics, account information, and service usage patterns.

---

2. 📊 Exploratory Data Analysis (EDA)

   * Summary statistics and visualization techniques were used to uncover patterns and relationships in the data.
   * Tools: seaborn, matplotlib, pandas profiling
   * Insights: Churn distribution, relationships between features like contract type, tenure, and churn.

---

3. 🧹 Data Preprocessing

   * Handling missing values
   * Label Encoding for categorical variables
   * Feature scaling (if required)
   * Converting TotalCharges to numeric
   * Saving encoders using pickle for future predictions

---

4. 🔀 Train-Test Split

   * Splitting the dataset into training and testing sets (typically 80/20 or 70/30)
   * Stratified sampling to preserve churn ratio across splits

---

5. 🤖 ML Models
   Trained and evaluated the following models:

   * Decision Tree Classifier
   * Random Forest Classifier
   * XGBoost Classifier

   ✅ Performance Evaluation:

   * Accuracy, Confusion Matrix, Classification Report
   * Cross-validation to validate model stability

   🏆 Best Model: Based on evaluation metrics, the best-performing model was selected and saved as a .pkl file using pickle.

---

6. 🔮 Unknown Data → Best Trained Model → Prediction

   * A pipeline is created to preprocess and encode new data
   * The best-trained model is loaded to make predictions on unseen data
   * The prediction outputs whether a customer is likely to churn or not

---

## 💡 Requirements

* Python 
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* pickle

---

## 📈 Results & Insights

* Customers with month-to-month contracts, no internet security, and high monthly charges were more likely to churn.
* Contract type and tenure are strong predictors of churn.
* Random Forest and XGBoost gave the best results in terms of balanced precision and recall.

---

