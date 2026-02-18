# src/stats_tests.py

import pandas as pd
from scipy.stats import ttest_ind


# =====================================================
# 1. SUPERLATIVE T-TEST
# =====================================================

def superlative_ttest(df):
    """
    Performs independent t-test comparing
    superlative usage between deceptive and truthful reviews.

    Based on your project code.
    """

    print("\n==============================")
    print(" Superlative Feature T-Test")
    print("==============================")

    deceptive_sup = df[df["deceptive"] == "deceptive"]["superlative_ratio"]
    truthful_sup = df[df["deceptive"] == "truthful"]["superlative_ratio"]

    t_stat, p_value = ttest_ind(
        deceptive_sup,
        truthful_sup,
        equal_var=False
    )

    print(f"Deceptive mean superlative ratio: {deceptive_sup.mean():.4f}")
    print(f"Truthful mean superlative ratio: {truthful_sup.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference ✅")
    else:
        print("Result: No significant difference ❌")

    return t_stat, p_value


# =====================================================
# 2. FIRST-PERSON PRONOUN T-TEST
# =====================================================

def pronoun_ttest(df):
    """
    Performs independent t-test comparing
    first-person pronoun usage between deceptive and truthful reviews.

    Based on your code:
        fp_per_100 feature
    """

    print("\n==============================")
    print(" First-Person Pronoun T-Test")
    print("==============================")

    deceptive_fp = df[df["deceptive"] == "deceptive"]["fp_per_100"]
    truthful_fp = df[df["deceptive"] == "truthful"]["fp_per_100"]

    t_stat, p_value = ttest_ind(
        deceptive_fp,
        truthful_fp,
        equal_var=False
    )

    print(f"Deceptive mean pronouns per 100 words: {deceptive_fp.mean():.4f}")
    print(f"Truthful mean pronouns per 100 words: {truthful_fp.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference ✅")
    else:
        print("Result: No significant difference ❌")

    return t_stat, p_value


# =====================================================
# 3. CORRELATION ANALYSIS
# =====================================================

def correlation_analysis(df):
    """
    Computes correlation between handcrafted features
    and deception label.

    Features:
      - superlative_ratio
      - fp_per_100
    """

    print("\n==============================")
    print(" Feature Correlation Analysis")
    print("==============================")

    # Convert

def run_full_statistical_tests(df):
    """
    Runs all statistical tests needed for thesis/paper:
      - Superlative t-test
      - Pronoun t-test
      - Correlation analysis
    """

    print("\n==========================================")
    print(" Running Full Statistical Hypothesis Tests")
    print("==========================================")

    superlative_ttest(df)
    pronoun_ttest(df)
    correlation_analysis(df)

    print("\nAll statistical tests completed.\n")
