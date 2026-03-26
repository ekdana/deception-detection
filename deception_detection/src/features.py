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
# 3. NEGATION FEATURE ENGINEERING
# =====================================================

def tokenize_text(text: str):
    """
    Lowercases and splits text into word tokens.
    Used internally by negation and hedging feature extractors.
    """
    if pd.isna(text):
        return []
    return re.findall(r"\b\w+\b", str(text).lower())


NEGATION_WORDS = [
    "not", "no", "never", "nothing", "nowhere", "nobody", "none",
    "neither", "nor", "n't", "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "don't", "doesn't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"
]


def count_negations(text: str) -> int:
    """
    Counts negation words in a review.
    """
    words = tokenize_text(text)
    return sum(word in NEGATION_WORDS for word in words)


def negation_per_100_words(text: str) -> float:
    """
    Normalized negation usage per 100 words.
    """
    words = tokenize_text(text)
    if len(words) == 0:
        return 0.0

    neg_count = count_negations(text)
    return (neg_count / len(words)) * 100


def extract_negation_features(df):
    """
    Adds negation_count and negation_per_100 columns to dataframe.
    """
    df["negation_count"] = df["text"].apply(count_negations)
    df["negation_per_100"] = df["text"].apply(negation_per_100_words)
    return df


def most_common_negations(text_series, top_n=10):
    """
    Returns the most frequent negation words in the dataset.
    """
    all_words = []

    for text in text_series:
        words = tokenize_text(text)
        all_words.extend([w for w in words if w in NEGATION_WORDS])

    return Counter(all_words).most_common(top_n)


# =====================================================
# 4. HEDGING FEATURE ENGINEERING
# =====================================================

HEDGING_WORDS = [
    # Modal verbs and adverbs of uncertainty
    "maybe", "probably", "possibly", "perhaps", "might", "may",
    "could", "would", "should", "seem", "seems", "seemed",
    "appear", "appears", "appeared",

    # Approximating expressions
    "about", "around", "approximately", "roughly", "nearly",
    "almost", "somewhat", "quite", "rather", "fairly",

    # Uncertainty markers
    "suppose", "guess", "think", "believe", "assume",
    "feel", "suspect", "doubt", "wonder",

    # Degree modifiers
    "pretty", "relatively", "reasonably"
]

MULTI_WORD_HEDGES = [
    "kind of", "sort of", "a bit", "a little"
]


def count_hedging(text: str) -> int:
    """
    Counts hedging expressions in a review.
    Includes both single-word and multi-word hedges.
    """
    if pd.isna(text):
        return 0

    text_lower = str(text).lower()
    count = 0

    # multi-word hedges
    for hedge in MULTI_WORD_HEDGES:
        count += text_lower.count(hedge)

    # single-word hedges
    words = tokenize_text(text)
    count += sum(word in HEDGING_WORDS for word in words)

    return count


def hedging_per_100_words(text: str) -> float:
    """
    Normalized hedging usage per 100 words.
    """
    words = tokenize_text(text)
    if len(words) == 0:
        return 0.0

    hedge_count = count_hedging(text)
    return (hedge_count / len(words)) * 100


def extract_hedging_features(df):
    """
    Adds hedging_count and hedging_per_100 columns to dataframe.
    """
    df["hedging_count"] = df["text"].apply(count_hedging)
    df["hedging_per_100"] = df["text"].apply(hedging_per_100_words)
    return df


def most_common_hedging(text_series, top_n=15):
    """
    Returns the most frequent hedging expressions in the dataset.
    Includes both single-word and multi-word hedges.
    """
    all_hedges = []

    for text in text_series:
        if pd.isna(text):
            continue

        text_lower = str(text).lower()

        # multi-word hedges
        for hedge in MULTI_WORD_HEDGES:
            count = text_lower.count(hedge)
            all_hedges.extend([hedge] * count)

        # single-word hedges
        words = tokenize_text(text)
        all_hedges.extend([w for w in words if w in HEDGING_WORDS])

    return Counter(all_hedges).most_common(top_n)


# =====================================================
# 5. FULL FEATURE PIPELINE
# =====================================================

def extract_all_features(df):
    """
    Runs all handcrafted feature extraction steps.
    Adds:
      - superlative_count
      - superlative_ratio
      - fp_count
      - fp_per_100
      - negation_count
      - negation_per_100
      - hedging_count
      - hedging_per_100
    """
    df = extract_superlative_features(df)
    df = extract_pronoun_features(df)
    df = extract_negation_features(df)
    df = extract_hedging_features(df)

    return df
