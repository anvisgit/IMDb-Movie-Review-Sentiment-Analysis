# IMDb Movie Review Sentiment Analysis Dashboard

## Overview
- **Streamlit-based interactive dashboard** for analyzing IMDb movie reviews.  
- Explore **text data**, **visualize sentiment distribution**, and **train multiple machine learning models** to predict sentiment.  
- View **model performance**, **feature importance**, **learning curves**, and **ROC curves** interactively.

---

## Features

- **Interactive Dashboard**
  - Built with **Streamlit**.
  - **Tabs:** Analysis, Logistic Regression, Random Forest, SVC.
  - Switch between tabs to **explore data** and **compare model performance**.

- **Text Exploration & WordClouds**
  - Visualizes **common words** in positive and negative reviews.
  - Understand **patterns in user sentiments**.
  - **Error WordCloud** highlights words in **misclassified reviews**.

- **Sentiment Distribution Visualization**
  - Sidebar shows **bar chart** for **positive vs. negative reviews**.
  - Provides a quick overview of the dataset’s **sentiment balance**.

- **Machine Learning Models**
  - Implements **Logistic Regression**, **Random Forest**, and **Linear SVC**.
  - **Training/Test split:** 70% / 30%.
  - **Evaluation metrics:**
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1 Score**
    - **Feature importance** (Logistic Regression & Random Forest)
    - **ROC Curve**

- **Confusion Matrix**
  - Displays **true positives**, **true negatives**, **false positives**, and **false negatives** for each model.

- **Learning Curves**
  - Shows how **model performance changes** with increasing training data.
  - Visualizes **training vs. test accuracy** to detect **overfitting** or **underfitting**.

- **Optimized for Performance**
  - Uses **@st.cache_data** and **@st.cache_resource** for **faster data loading** and **model reuse**.

---

## Python Stack
- **Streamlit** – interactive dashboard
- **Pandas & NumPy** – data manipulation
- **Matplotlib** – visualizations
- **Scikit-learn** – machine learning models
- **WordCloud** – word cloud generation

---
