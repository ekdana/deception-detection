# src/predict.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =====================================================
# Load Fine-Tuned Model
# =====================================================

def load_model(model_path="saved_bert_model"):
    """
    Loads your fine-tuned BERT model and tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    return model, tokenizer


# =====================================================
# Predict Function
# =====================================================

def predict_text(text, model, tokenizer):
    """
    Predicts whether input text is deceptive or truthful.

    Returns:
      - label
      - trustworthiness score
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probs = torch.softmax(logits, dim=1).numpy()[0]

    truthful_prob = probs[0]
    deceptive_prob = probs[1]

    # Prediction
    predicted_class = probs.argmax()

    if predicted_class == 1:
        label = "DECEPTIVE ❌"
        trust_score = truthful_prob
    else:
        label = "TRUTHFUL ✅"
        trust_score = truthful_prob

    return label, trust_score, deceptive_prob


# =====================================================
# Interactive Terminal System
# =====================================================

def interactive_system():
    """
    Terminal-based deception detection system.
    """

    print("\n==============================================")
    print("   Deceptive Review Detection System (BERT)")
    print("==============================================")

    model, tokenizer = load_model()

    while True:
        print("\nEnter a review text (or type 'exit' to quit):\n")

        user_text = input("> ")

        if user_text.lower() == "exit":
            print("\nExiting system. Goodbye!")
            break

        label, trust_score, deception_score = predict_text(
            user_text,
            model,
            tokenizer
        )

        print("\n------------------------------")
        print(" Prediction Result")
        print("------------------------------")
        print(f"Classification: {label}")

        print(f"Trustworthiness Score: {trust_score:.2f}")
        print(f"Deception Probability: {deception_score:.2f}")
        print("------------------------------")


# =====================================================
# Run Script
# =====================================================

if __name__ == "__main__":
    interactive_system()
