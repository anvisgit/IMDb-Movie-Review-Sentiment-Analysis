import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import roc_curve, auc
from wordcloud import WordCloud

@st.cache_data
def load():
    df = pd.read_csv("IMDB Dataset.csv")
    df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
    return df

@st.cache_resource
def prepare(df):
    x, y = df['review'], df['sentiment']
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=3)
    vectorizer = TfidfVectorizer(max_features=5000)
    trainvecx = vectorizer.fit_transform(trainx)
    testvecx = vectorizer.transform(testx)
    return vectorizer, trainvecx, testvecx, trainy, testy, trainx, testx
st.set_page_config(page_title='Movie Review', page_icon="👾", layout='wide')
st.header("IMDb Movie Review")
st.markdown("---")
st.sidebar.header("")
st.write(
    "Streamlit-based interactive dashboard that analyzes IMDb movie reviews, "
    "providing insights into sentiment distribution and review characteristics. "
    "It also allows users to train machine learning models and predict the sentiment "
    "of new reviews in real-time."
)
uploaded = st.button("DRUMROLLS PLEASE")
tab0, tab1, tab2 , tab3= st.tabs(["Analysis","Logistic Regression", "Random Forest", "SVC"])

if uploaded:
    df = load()
    if 'review' in df.columns and 'sentiment' in df.columns:
        st.sidebar.subheader("Sentiment Distribution")
        st.sidebar.write(df['sentiment'].value_counts())
        st.sidebar.bar_chart(df['sentiment'].value_counts())
        st.sidebar.caption("Anvi Wadhwa")
        st.sidebar.caption("24BCE5054")

        vectorizer, trainvecx, testvecx, trainy, testy, trainx, testx = prepare(df)

        with tab0:
            st.header("Text explorer")
            st.subheader("WordCloud")
            good = " ".join(df[df["sentiment"]==1]["review"])
            bad = " ".join(df[df["sentiment"]==0]["review"])
            text = good + " " + bad
            wc = WordCloud(width=700, height=600, colormap="Greens", background_color="black").generate(text)
            st.image(wc.to_array())

            st.subheader("Error Analysis")
            model = LogisticRegression(max_iter=1000, solver='liblinear')
            model.fit(trainvecx, trainy)
            ypred = model.predict(testvecx)
            
            errors = pd.DataFrame({
                "review": testx.reset_index(drop=True),
                "true_sentiment": testy.reset_index(drop=True),
                "predicted_sentiment": ypred,
                
            })
            errors = errors[errors['true_sentiment'] != errors['predicted_sentiment']]
            st.write(f"Number of misclassified reviews: {errors.shape[0]}")
            
            st.dataframe(errors.head(10))
            error_text = " ".join(errors['review'].head(1000))
            wc_err = WordCloud(width=700, height=400, colormap="Reds", background_color="black").generate(error_text)
            st.image(wc_err.to_array())

        with tab1:
            st.subheader("LOGISTIC REGRESSION")
            model = LogisticRegression(max_iter=1000, solver='liblinear')
            model.fit(trainvecx, trainy)
            ypred = model.predict(testvecx)
            st.write("Accuracy:", accuracy_score(testy, ypred))
            st.write("Precision:", precision_score(testy, ypred, average='weighted'))
            st.write("Recall:", recall_score(testy, ypred, average='weighted'))
            st.write("F1 Score:", f1_score(testy, ypred, average='weighted'))
            c1,c2=st.columns(2)
            with c1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(testy, ypred)
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Greens, ax=ax, colorbar=False)
                st.pyplot(fig)
            with c2:
                st.subheader("Learning Curve")
                train_sizes, train_scores, test_scores = learning_curve(model, trainvecx, trainy, cv=2, train_sizes=np.linspace(0.1, 1.0, 3), scoring='accuracy')
                train_mean = np.mean(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                fig2, ax2 = plt.subplots()
                ax2.plot(train_sizes, train_mean, label='Train Accuracy', color='green')
                ax2.plot(train_sizes, test_mean, label='Test Accuracy', color='green')
                ax2.set_xlabel("Training Samples")
                ax2.set_ylabel("Accuracy")
                ax2.set_title("Learning Curve")
                ax2.legend()
                st.pyplot(fig2)
            st.subheader("Feature Importance")
            importance = np.abs(model.coef_[0])
            indices = np.argsort(importance)[-20:]
            st.bar_chart(pd.Series(importance[indices], index=np.array(vectorizer.get_feature_names_out())[indices]))
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(testvecx)[:, 1]
            else:
                y_scores = model.decision_function(testvecx)
            fpr, tpr, _ = roc_curve(testy, y_scores)
            roc_auc = auc(fpr, tpr)
            st.markdown("---")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color="green", label=f"AUC = {roc_auc:.2f}")
            ax.plot([0,1],[0,1],'--',color='gray')
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
        with tab2:
            st.subheader("RANDOM FOREST")
            model = RandomForestClassifier(n_estimators=20,max_depth=15,max_features='sqrt',n_jobs=-1,random_state=3)
            model.fit(trainvecx, trainy)
            model.fit(trainvecx, trainy)
            ypred = model.predict(testvecx)
            st.write("Accuracy:", accuracy_score(testy, ypred))
            st.write("Precision:", precision_score(testy, ypred, average='weighted'))
            st.write("Recall:", recall_score(testy, ypred, average='weighted'))
            st.write("F1 Score:", f1_score(testy, ypred, average='weighted'))
            c1,c2=st.columns(2)
            with c1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(testy, ypred)
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Greens, ax=ax, colorbar=False)
                st.pyplot(fig)
            with c2:
                st.subheader("Learning Curve")
                train_sizes, train_scores, test_scores = learning_curve(model, trainvecx, trainy, cv=2, train_sizes=np.linspace(0.1, 1.0, 3), scoring='accuracy')
                train_mean = np.mean(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                fig2, ax2 = plt.subplots()
                ax2.plot(train_sizes, train_mean, label='Train Accuracy', color='green')
                ax2.plot(train_sizes, test_mean, label='Test Accuracy', color='green')
                ax2.set_xlabel("Training Samples")
                ax2.set_ylabel("Accuracy")
                ax2.set_title("Learning Curve")
                ax2.legend()
                st.pyplot(fig2)
            st.markdown("---")
            importance = model.feature_importances_
            indices = np.argsort(importance)[-20:][::-1]  # descending order
            st.bar_chart(pd.Series(importance[indices], index=np.array(vectorizer.get_feature_names_out())[indices]))

            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(testvecx)[:, 1]
            else:
                y_scores = model.decision_function(testvecx)
            fpr, tpr, _ = roc_curve(testy, y_scores)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color="green", label=f"AUC = {roc_auc:.2f}")
            ax.plot([0,1],[0,1],'--',color='gray')
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
        with tab3:
            st.subheader("SVC")
            model = LinearSVC(C=1.5, max_iter=5000, random_state=42)
            model.fit(trainvecx, trainy)
            ypred = model.predict(testvecx)
            st.write("Accuracy :", accuracy_score(testy, ypred))
            st.write("Precision:", precision_score(testy, ypred, average='weighted'))
            st.write("Recall Score:", recall_score(testy, ypred, average='weighted'))
            st.write("F1 Score:", f1_score(testy, ypred, average='weighted'))
            c1,c2=st.columns(2)
            with c1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(testy, ypred)
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Greens, ax=ax, colorbar=False)
                st.pyplot(fig)
            with c2:
                st.subheader("Learning Curve")
                train_sizes, train_scores, test_scores = learning_curve(model, trainvecx, trainy, cv=2, train_sizes=np.linspace(0.1, 1.0, 3), scoring='accuracy')
                train_mean = np.mean(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                fig2, ax2 = plt.subplots()
                ax2.plot(train_sizes, train_mean, label='Train Accuracy', color='green')
                ax2.plot(train_sizes, test_mean, label='Test Accuracy', color='green')
                ax2.set_xlabel("Training Samples")
                ax2.set_ylabel("Accuracy")
                ax2.set_title("Learning Curve")
                ax2.legend()
                st.pyplot(fig2)
        
