# src/classical_models.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


#x =====================================================
# 1. TF-IDF VECTORIZATION
# =====================================================

def build_tfidf(max_features=2000):
    """
    Creates TF-IDF vectorizer for text classification.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )

    return vectorizer


# =====================================================
# 2. LOGISTIC REGRESSION MODEL
# =====================================================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Logistic Regression classifier.
    Based on your implementation.
    """

    print("\n==============================")
    print(" Logistic Regression Results")
    print("==============================")

    vectorizer = build_tfidf()

    # Vectorize text
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vec, y_train)

    # Predictions
    y_pred = model.predict(X_test_vec)

    # Report
    print(classification_report(y_test, y_pred))

    return model, vectorizer


# =====================================================
# 3. SUPPORT VECTOR CLASSIFIER (SVC)
# =====================================================

def train_svc(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Support Vector Machine classifier.
    Based on your implementation.
    """

    print("\n==============================")
    print(" Support Vector Classifier Results")
    print("==============================")

    vectorizer = build_tfidf()

    # Vectorize text
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train_vec, y_train)

    # Predictions
    y_pred = model.predict(X_test_vec)

    # Report
    print(classification_report(y_test, y_pred))

    return model, vectorizer


# =====================================================
# 4. CONFUSION MATRIX VISUALIZATION
# =====================================================

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    """
    Simple confusion matrix plot for thesis/paper.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()
