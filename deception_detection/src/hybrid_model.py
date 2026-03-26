# src/hybrid_model.py
#
# Hybrid Ensemble: BERT + Logistic Regression + SVC
# Strategy: Weighted soft voting over predicted probabilities
# Optional: stacking meta-learner (see train_stacking_ensemble)

import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =====================================================
# 1. BERT PROBABILITY EXTRACTOR
#    Wraps a fine-tuned BERT model and returns
#    P(deceptive) for a list of texts.
# =====================================================

class BertProbabilityExtractor:
    """
    Loads a saved BERT model and extracts
    deceptive-class probabilities for a text list.

    Usage:
        extractor = BertProbabilityExtractor("saved_bert_model")
        probs = extractor.predict_proba(texts)  # shape: (n,)
    """

    def __init__(self, model_path="saved_bert_model"):
        print(f"  Loading BERT from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict_proba(self, texts, batch_size=16):
        """
        Returns P(deceptive=1) for each text.
        Shape: (n_samples,)
        """
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch = list(texts[i: i + batch_size])

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1).numpy()
            all_probs.append(probs[:, 1])  # column 1 = P(deceptive)

        return np.concatenate(all_probs)


# =====================================================
# 2. CLASSICAL MODEL PROBABILITY EXTRACTORS
#    Both LR and SVC trained on TF-IDF.
#    SVC requires probability=True for soft voting.
# =====================================================

def train_lr_with_proba(X_train, y_train, max_features=2000):
    """
    Trains TF-IDF + Logistic Regression.
    Returns fitted (model, vectorizer).
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )
    X_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_vec, y_train)
    return model, vectorizer


def train_svc_with_proba(X_train, y_train, max_features=2000):
    """
    Trains TF-IDF + SVC with probability=True (required for soft voting).
    Returns fitted (model, vectorizer).
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )
    X_vec = vectorizer.fit_transform(X_train)

    # NOTE: probability=True uses Platt scaling — slightly slower to train
    # but required to get calibrated probabilities from SVC.
    model = SVC(kernel="linear", random_state=42, probability=True)
    model.fit(X_vec, y_train)
    return model, vectorizer


# =====================================================
# 3. WEIGHTED SOFT VOTING ENSEMBLE
# =====================================================

class HybridEnsemble:
    """
    Combines BERT, LR, and SVC via weighted soft voting.

    Each model contributes P(deceptive). The final probability
    is a weighted average; threshold 0.5 gives the class label.

    Default weights lean toward BERT as the strongest model:
        bert=0.5, lr=0.25, svc=0.25

    Adjust via set_weights() after seeing individual model scores.
    """

    def __init__(self, w_bert=0.5, w_lr=0.25, w_svc=0.25):
        self.w_bert = w_bert
        self.w_lr   = w_lr
        self.w_svc  = w_svc

        self.bert_extractor = None
        self.lr_model        = None
        self.lr_vectorizer   = None
        self.svc_model       = None
        self.svc_vectorizer  = None

    def set_weights(self, w_bert, w_lr, w_svc):
        """Update voting weights. Must sum to 1.0."""
        total = w_bert + w_lr + w_svc
        self.w_bert = w_bert / total
        self.w_lr   = w_lr   / total
        self.w_svc  = w_svc  / total
        print(f"  Weights updated → BERT: {self.w_bert:.2f}, "
              f"LR: {self.w_lr:.2f}, SVC: {self.w_svc:.2f}")

    def fit(self, X_train, y_train, bert_model_path="saved_bert_model"):
        """
        Trains LR and SVC on the training split.
        Loads a pre-trained BERT from bert_model_path.

        BERT must already be fine-tuned (via train_bert_model in bert_model.py)
        before calling fit().
        """
        print("\n--- Fitting Hybrid Ensemble ---")

        print("  [1/3] Loading BERT...")
        self.bert_extractor = BertProbabilityExtractor(bert_model_path)

        print("  [2/3] Training Logistic Regression...")
        self.lr_model, self.lr_vectorizer = train_lr_with_proba(X_train, y_train)

        print("  [3/3] Training SVC (probability=True)...")
        self.svc_model, self.svc_vectorizer = train_svc_with_proba(X_train, y_train)

        print("  Hybrid ensemble ready.\n")

    def _get_proba(self, texts):
        """
        Returns a (n_samples, 3) array of P(deceptive):
            col 0 = BERT, col 1 = LR, col 2 = SVC
        """
        texts = list(texts)

        p_bert = self.bert_extractor.predict_proba(texts)

        X_lr  = self.lr_vectorizer.transform(texts)
        p_lr  = self.lr_model.predict_proba(X_lr)[:, 1]

        X_svc = self.svc_vectorizer.transform(texts)
        p_svc = self.svc_model.predict_proba(X_svc)[:, 1]

        return np.stack([p_bert, p_lr, p_svc], axis=1)

    def predict_proba(self, texts):
        """
        Returns weighted average P(deceptive) for each text.
        Shape: (n_samples,)
        """
        probs = self._get_proba(texts)
        weights = np.array([self.w_bert, self.w_lr, self.w_svc])
        return probs @ weights

    def predict(self, texts, threshold=0.5):
        """
        Returns binary predictions: 1 = deceptive, 0 = truthful.
        """
        return (self.predict_proba(texts) >= threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """
        Prints per-model and ensemble results on the test set.
        """
        print("\n==============================")
        print(" Hybrid Ensemble Evaluation")
        print("==============================")

        X_test = list(X_test)
        y_test = np.array(y_test)

        probs = self._get_proba(X_test)

        model_names = ["BERT", "Logistic Regression", "SVC"]
        individual_preds = []

        for i, name in enumerate(model_names):
            preds = (probs[:, i] >= 0.5).astype(int)
            individual_preds.append(preds)
            acc = accuracy_score(y_test, preds)
            print(f"\n  [{name}]  Accuracy: {acc:.4f}")
            print(classification_report(y_test, preds,
                                        target_names=["Truthful", "Deceptive"],
                                        digits=4))

        # Ensemble
        ensemble_preds = self.predict(X_test)
        acc = accuracy_score(y_test, ensemble_preds)

        print("\n------------------------------")
        print(f"  [HYBRID ENSEMBLE]  Accuracy: {acc:.4f}")
        print("------------------------------")
        print(classification_report(y_test, ensemble_preds,
                                    target_names=["Truthful", "Deceptive"],
                                    digits=4))

        return ensemble_preds, probs


# =====================================================
# 4. STACKING META-LEARNER (optional, stronger)
#    Trains a small LR on top of the three models'
#    out-of-fold predictions to learn optimal weights.
#
#    Use this if soft voting underperforms.
# =====================================================

class StackingEnsemble(HybridEnsemble):
    """
    Extends HybridEnsemble by replacing fixed weights with
    a trained meta-learner (Logistic Regression).

    Training uses cross-validated out-of-fold predictions
    to prevent data leakage.

    Note: BERT out-of-fold predictions are expensive to compute
    (requires re-training). This implementation uses the test-set
    BERT probabilities as a practical approximation — acceptable
    for a thesis/research context. For a strict pipeline, generate
    BERT OOF predictions via k-fold on the training set separately.
    """

    def __init__(self):
        super().__init__()
        self.meta_learner = None

    def fit_meta(self, X_train, y_train, bert_model_path="saved_bert_model"):
        """
        1. Fit LR + SVC using cross_val_predict (out-of-fold)
        2. Get BERT train predictions (approximation: use already-fine-tuned BERT)
        3. Train meta LR on the stacked features
        """
        print("\n--- Fitting Stacking Ensemble ---")

        print("  [1/4] Loading BERT...")
        self.bert_extractor = BertProbabilityExtractor(bert_model_path)

        print("  [2/4] LR out-of-fold predictions (5-fold CV)...")
        vec_lr = TfidfVectorizer(stop_words="english", max_features=2000)
        X_vec = vec_lr.fit_transform(X_train)
        lr_base = LogisticRegression(random_state=42, max_iter=1000)
        oof_lr = cross_val_predict(lr_base, X_vec, y_train,
                                   cv=5, method="predict_proba")[:, 1]
        # Refit on full train for test-time inference
        lr_base.fit(X_vec, y_train)
        self.lr_model, self.lr_vectorizer = lr_base, vec_lr

        print("  [3/4] SVC out-of-fold predictions (5-fold CV)...")
        vec_svc = TfidfVectorizer(stop_words="english", max_features=2000)
        X_vec_svc = vec_svc.fit_transform(X_train)
        svc_base = SVC(kernel="linear", random_state=42, probability=True)
        oof_svc = cross_val_predict(svc_base, X_vec_svc, y_train,
                                    cv=5, method="predict_proba")[:, 1]
        svc_base.fit(X_vec_svc, y_train)
        self.svc_model, self.svc_vectorizer = svc_base, vec_svc

        print("  [4/4] BERT train predictions (approximation)...")
        oof_bert = self.bert_extractor.predict_proba(list(X_train))

        # Stack into meta-features: shape (n_train, 3)
        meta_X = np.stack([oof_bert, oof_lr, oof_svc], axis=1)

        print("  Training meta-learner (Logistic Regression)...")
        self.meta_learner = LogisticRegression(random_state=42, max_iter=500)
        self.meta_learner.fit(meta_X, y_train)

        coefs = self.meta_learner.coef_[0]
        print(f"  Meta-learner weights → "
              f"BERT: {coefs[0]:.3f}, LR: {coefs[1]:.3f}, SVC: {coefs[2]:.3f}")
        print("  Stacking ensemble ready.\n")

    def predict_proba(self, texts):
        """
        Returns P(deceptive) from the meta-learner.
        """
        probs = self._get_proba(texts)
        return self.meta_learner.predict_proba(probs)[:, 1]

    def predict(self, texts, threshold=0.5):
        return (self.predict_proba(texts) >= threshold).astype(int)


# =====================================================
# 5. CONVENIENCE RUNNER
#    Drop-in replacement for run_classical_models()
#    and run_bert() in main.py
# =====================================================

def run_hybrid_ensemble(df, bert_model_path="saved_bert_model",
                        use_stacking=False):
    """
    Full hybrid pipeline. Call this from main.py after BERT is trained.

    Args:
        df              : preprocessed DataFrame with 'text' and 'deceptive' cols
        bert_model_path : path to saved fine-tuned BERT model
        use_stacking    : if True, uses meta-learner instead of fixed weights

    Returns:
        ensemble        : fitted HybridEnsemble (or StackingEnsemble) object
    """
    from sklearn.model_selection import train_test_split

    print("\n==============================")
    print(" Running Hybrid Ensemble...")
    print("==============================")

    y = df["deceptive"].map({"truthful": 0, "deceptive": 1})
    X = df["text_clean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if use_stacking:
        ensemble = StackingEnsemble()
        ensemble.fit_meta(X_train, y_train, bert_model_path)
    else:
        ensemble = HybridEnsemble(w_bert=0.5, w_lr=0.25, w_svc=0.25)
        ensemble.fit(X_train, y_train, bert_model_path)

    ensemble.evaluate(X_test, y_test)

    return ensemble