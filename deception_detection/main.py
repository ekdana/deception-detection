# main.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Import our modules
from src.preprocess import preprocess_text
from src.features import extract_all_features
from src.stats_tests import run_full_statistical_tests
from src.classical_models import train_logistic_regression, train_svc
from src.bert_model import train_bert_model


# =====================================================
# 1. LOAD DATASET
# =====================================================

def load_data():
    """
    Loads the deceptive opinion dataset.
    """

    print("\n==============================")
    print(" Loading Dataset...")
    print("==============================")

    df = pd.read_csv("data/deceptive-opinion.csv")

    # Rename columns (same as your original code)
    df.columns = ["deceptive", "hotel", "polarity", "source", "text"]

    print("Dataset loaded successfully!")
    print("Total samples:", len(df))

    return df


# =====================================================
# 2. PREPROCESS TEXT
# =====================================================

def clean_text(df):
    """
    Applies preprocessing to raw review text.
    """

    print("\n==============================")
    print(" Text Preprocessing...")
    print("==============================")

    df["text"] = df["text"].apply(preprocess_text)

    print("Text preprocessing completed.")

    return df


# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

def add_features(df):
    """
    Extract handcrafted linguistic features:
    - superlatives
    - first-person pronouns
    """

    print("\n==============================")
    print(" Extracting Linguistic Features...")
    print("==============================")

    df = extract_all_features(df)

    print("Features added successfully!")
    print(df[["superlative_ratio", "fp_per_100"]].head())

    return df


# =====================================================
# 4. STATISTICAL HYPOTHESIS TESTING
# =====================================================

def run_statistics(df):
    """
    Runs t-tests + correlation analysis.
    """

    run_full_statistical_tests(df)


# =====================================================
# 5. BASELINE CLASSICAL ML MODELS
# =====================================================

def run_classical_models(df):
    """
    Runs TF-IDF + Logistic Regression and SVC.
    """

    print("\n==============================")
    print(" Running Classical ML Models...")
    print("==============================")

    # Encode labels
    y = df["deceptive"].map({"truthful": 0, "deceptive": 1})
    X = df["text"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    train_logistic_regression(X_train, X_test, y_train, y_test)

    # Support Vector Classifier
    train_svc(X_train, X_test, y_train, y_test)


# =====================================================
# 6. BERT FINE-TUNING
# =====================================================

def run_bert(df):
    """
    Fine-tunes BERT model for deception detection.
    """

    print("\n==============================")
    print(" Running BERT Fine-Tuning...")
    print("==============================")

    # Labels
    df["label"] = df["deceptive"].map({"truthful": 0, "deceptive": 1})

    # Train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Train BERT
    train_bert_model(train_texts, test_texts, train_labels, test_labels)


# =====================================================
# 7. MAIN PIPELINE
# =====================================================

def main():
    """
    Full Deception Detection Research Pipeline.
    """

    # Step 1: Load dataset
    df = load_data()

    # Step 2: Preprocess text
    df = clean_text(df)

    # Step 3: Feature extraction
    df = add_features(df)

    # Step 4: Statistical tests (hypothesis validation)
    run_statistics(df)

    # Step 5: Classical baseline ML models
    run_classical_models(df)

    # Step 6: BERT fine-tuning
    run_bert(df)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
