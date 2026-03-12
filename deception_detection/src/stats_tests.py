# src/stats_tests.py

import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


# =====================================================
# 0. EFFECT SIZE FUNCTION
# =====================================================

def cohens_d(x, y):
    """
    Computes Cohen's d effect size.

    Interpretation:
    0.2 = small
    0.5 = medium
    0.8 = large
    """

    nx = len(x)
    ny = len(y)

    pooled_std = (((nx - 1) * x.var() + (ny - 1) * y.var()) / (nx + ny - 2)) ** 0.5

    return (x.mean() - y.mean()) / pooled_std


# =====================================================
# 1. SUPERLATIVE T-TEST
# =====================================================

def superlative_ttest(df):

    print("\n==============================")
    print(" Superlative Feature T-Test")
    print("==============================")

    deceptive_sup = df[df["deceptive"] == "deceptive"]["superlative_ratio"]
    truthful_sup = df[df["deceptive"] == "truthful"]["superlative_ratio"]

    t_stat, p_value = ttest_ind(deceptive_sup, truthful_sup, equal_var=False)

    print(f"Deceptive mean superlative ratio: {deceptive_sup.mean():.4f}")
    print(f"Truthful mean superlative ratio: {truthful_sup.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    return t_stat, p_value


# =====================================================
# 2. FIRST-PERSON PRONOUN T-TEST + BOX PLOT
# =====================================================

def pronoun_ttest(df):

    print("\n==============================")
    print(" First-Person Pronoun Deep Test")
    print("==============================")

    deceptive_fp = df[df["deceptive"] == "deceptive"]["fp_per_100"]
    truthful_fp = df[df["deceptive"] == "truthful"]["fp_per_100"]

    # Means
    print(f"Deceptive mean pronouns per 100 words: {deceptive_fp.mean():.4f}")
    print(f"Truthful mean pronouns per 100 words : {truthful_fp.mean():.4f}")

    # T-test
    t_stat, p_value = ttest_ind(deceptive_fp, truthful_fp, equal_var=False)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    # Effect size
    d = cohens_d(deceptive_fp, truthful_fp)
    print(f"Cohen’s d effect size: {d:.4f}")

    # Boxplot saved for paper
    plt.figure(figsize=(6, 4))
    plt.boxplot([truthful_fp, deceptive_fp], labels=["Truthful", "Deceptive"])
    plt.title("Pronoun Usage Distribution")
    plt.ylabel("Pronouns per 100 words")
    plt.savefig("pronoun_boxplot.png")
    print("Saved pronoun_boxplot.png")



    return t_stat, p_value, d


# =====================================================
# 3. NEGATION T-TEST + BOX PLOT
# =====================================================

def negation_ttest(df):

    print("\n==============================")
    print(" Negation Deep Test")
    print("==============================")

    deceptive_neg = df[df["deceptive"] == "deceptive"]["negation_per_100"]
    truthful_neg = df[df["deceptive"] == "truthful"]["negation_per_100"]

    print(f"Deceptive mean negations per 100 words: {deceptive_neg.mean():.4f}")
    print(f"Truthful mean negations per 100 words : {truthful_neg.mean():.4f}")

    t_stat, p_value = ttest_ind(deceptive_neg, truthful_neg, equal_var=False)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    d = cohens_d(deceptive_neg, truthful_neg)
    print(f"Cohen’s d effect size: {d:.4f}")

    plt.figure(figsize=(6,4))
    plt.boxplot([truthful_neg, deceptive_neg], labels=["Truthful","Deceptive"])
    plt.title("Negation Usage Distribution")
    plt.ylabel("Negations per 100 words")
    plt.savefig("negation_boxplot.png")

    return t_stat, p_value, d

# =====================================================
# 4. HEDGING T-TEST + BOX PLOT
# =====================================================

def hedging_ttest(df):

    print("\n==============================")
    print(" Hedging Deep Test")
    print("==============================")

    deceptive_hedge = df[df["deceptive"] == "deceptive"]["hedging_per_100"]
    truthful_hedge = df[df["deceptive"] == "truthful"]["hedging_per_100"]

    print(f"Deceptive mean hedging per 100 words: {deceptive_hedge.mean():.4f}")
    print(f"Truthful mean hedging per 100 words : {truthful_hedge.mean():.4f}")

    t_stat, p_value = ttest_ind(deceptive_hedge, truthful_hedge, equal_var=False)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    d = cohens_d(deceptive_hedge, truthful_hedge)
    print(f"Cohen’s d effect size: {d:.4f}")

    plt.figure(figsize=(6,4))
    plt.boxplot([truthful_hedge, deceptive_hedge], labels=["Truthful","Deceptive"])
    plt.title("Hedging Usage Distribution")
    plt.ylabel("Hedging per 100 words")
    plt.savefig("hedging_boxplot.png")

    return t_stat, p_value, d

# =====================================================
# 5. CORRELATION ANALYSIS
# =====================================================

def correlation_analysis(df):

    print("\n==============================")
    print(" Feature Correlation Analysis")
    print("==============================")

    df["label"] = df["deceptive"].map({"truthful": 0, "deceptive": 1})

    sup_corr = df["superlative_ratio"].corr(df["label"])
    fp_corr = df["fp_per_100"].corr(df["label"])
    neg_corr = df["negation_per_100"].corr(df["label"])
    hedge_corr = df["hedging_per_100"].corr(df["label"])

    print(f"Correlation(superlatives, deception): {sup_corr:.4f}")
    print(f"Correlation(pronouns, deception): {fp_corr:.4f}")
    print(f"Correlation(negations, deception): {neg_corr:.4f}")
    print(f"Correlation(hedging, deception): {hedge_corr:.4f}")

    return sup_corr, fp_corr, neg_corr, hedge_corr


# =====================================================
# 6. FULL REPORT
# =====================================================

def run_full_statistical_tests(df):

    print("\n==========================================")
    print(" Running Full Statistical Hypothesis Tests")
    print("==========================================")

    superlative_ttest(df)
    pronoun_ttest(df)
    correlation_analysis(df)

    print("\nAll statistical tests completed.\n")
