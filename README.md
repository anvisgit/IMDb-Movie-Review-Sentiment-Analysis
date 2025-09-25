## Features

- **Interactive Dashboard**
  - Built with Streamlit.
  - tabs: Analysis, Logistic Regression, Random Forest, SVC and ensemble.
  - Can switch between tabs to explore data and model performance.

- **Text Exploration & WordClouds**
  - Visualizes the most common words in positive and negative reviews.
  - Helps understand patterns in user sentiments.
  - WordClouds are dynamically generated based on the dataset.

- **Sentiment Distribution Visualization**
  - Displays a **bar chart**  showing positive vs. negative reviews.
  - Provides a quick overview of the dataset’s sentiment balance.

- **Machine Learning Models**
  - **Logistic Regression, Random Forest, and SVC** implemented for sentiment classification.
  - Each model is trained on 70% of the dataset and tested on 30%.
  - - **Machine Learning Models**
  - Implements **Logistic Regression**, **Random Forest**, and **Linear SVM** for sentiment classification.
  - Displays **performance metrics**:
    - **Logistic Regression**: Accuracy 89%
    - **Random Forest**: Accuracy 80%
    - **Linear SVM**: Accuracy 88%
  - Easy comparison of model performance to select the best predictor.
  - evaluation metrics shown:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - feature importance
    - roc curve
    - 

- **Confusion Matrix**
  - Displays the confusion matrix for each model.
  - Helps evaluate the model’s predictions for true positives, true negatives, false positives, and false negatives.

- **Learning Curves**
  - Shows how model performance changes with increasing training data.
  - Visualizes training vs. test accuracy to detect overfitting or underfitting.

- **Optimized for Performance**
  - Uses `@st.cache_data` and `@st.cache_resource` for faster data loading and model reuse.
  

---
