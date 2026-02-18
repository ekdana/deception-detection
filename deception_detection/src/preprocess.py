# src/preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------
# 1. Download NLTK resources (only first time)
# ---------------------------------------------------

def download_nltk_resources():
    """
    Downloads necessary NLTK datasets.
    Run this once before preprocessing.
    """
    nltk.download("stopwords")
    nltk.download("wordnet")


# ---------------------------------------------------
# 2. Initialize tools
# ---------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------
# 3. Main preprocessing function
# ---------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes raw review text.

    Steps:
    1. Lowercase
    2. Remove HTML tags
    3. Remove punctuation/numbers
    4. Remove stopwords
    5. Lemmatize words

    Returns cleaned text.
    """

    if text is None:
        return ""

    # Convert to string + lowercase
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-letter characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize + remove stopwords + lemmatize
    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)
