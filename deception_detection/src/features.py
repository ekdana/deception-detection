# src/features.py

import re
import pandas as pd
from collections import Counter


# =====================================================
# 1. SUPERLATIVE FEATURE ENGINEERING
# =====================================================

# базовый список (как в вашем проекте)
SUPERLATIVE_WORDS = [
    "best", "greatest", "most", "least", "finest", "nicest", "worst",
    "amazing", "wonderful", "perfect", "superb", "fantastic", "awesome",
    "incredible", "outstanding", "unbelievable"
]


def count_superlatives(text: str) -> int:
    """
    Counts the number of superlative/exaggeration words in a review.
    Example:
        "This is the best hotel ever!" → 1
    """

    if pd.isna(text):
        return 0

    words = re.findall(r"\b\w+\b", str(text).lower())
    return sum(word in SUPERLATIVE_WORDS for word in words)


def superlative_ratio(text: str) -> float:
    """
    Ratio of superlatives to total words.
    Helps normalize for review length.
    """

    if pd.isna(text):
        return 0.0

    words = re.findall(r"\b\w+\b", str(text).lower())
    if len(words) == 0:
        return 0.0

    return count_superlatives(text) / len(words)


def extract_superlative_features(df):
    """
    Adds superlative_count and superlative_ratio columns to dataframe.
    """

    df["superlative_count"] = df["text"].apply(count_superlatives)
    df["superlative_ratio"] = df["text"].apply(superlative_ratio)

    return df


def most_common_superlatives(text_series, top_n=10):
    """
    Returns the most frequent superlative words in the dataset.
    Useful for analysis section in paper.
    """

    all_words = []

    for text in text_series:
        if pd.isna(text):
            continue

        words = re.findall(r"\b\w+\b", str(text).lower())
        all_words.extend([w for w in words if w in SUPERLATIVE_WORDS])

    return Counter(all_words).most_common(top_n)


# =====================================================
# 2. FIRST-PERSON PRONOUN FEATURE ENGINEERING
# =====================================================

FIRST_PERSON_PRONOUNS = [
    "i", "me", "my", "mine",
    "we", "us", "our", "ours"
]


def count_first_person(text: str) -> int:
    """
    Counts first-person pronouns in the review.
    Hypothesis:
        deceptive reviews may use more self-reference.
    """

    if pd.isna(text):
        return 0

    words = str(text).lower().split()
    return sum(word in FIRST_PERSON_PRONOUNS for word in words)


def first_person_per_100_words(text: str) -> float:
    """
    Normalized pronoun usage per 100 words.
    """

    if pd.isna(text):
        return 0.0

    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0

    fp_count = count_first_person(text)
    return (fp_count / len(words)) * 100


def extract_pronoun_features(df):
    """
    Adds fp_count and fp_per_100 columns to dataframe.
    """

    df["fp_count"] = df["text"].apply(count_first_person)
    df["fp_per_100"] = df["text"].apply(first_person_per_100_words)

    return df


# =====================================================
# 3. FULL FEATURE PIPELINE
# =====================================================

def extract_all_features(df):
    """
    Runs all handcrafted feature extraction steps.
    Adds:
      - superlative_count
      - superlative_ratio
      - fp_count
      - fp_per_100
    """

    df = extract_superlative_features(df)
    df = extract_pronoun_features(df)

    return df
