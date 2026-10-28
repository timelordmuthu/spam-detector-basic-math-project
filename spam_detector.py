"""
Probabilistic SMS Spam Detection
BAMAT207 Probability and Statistics - Project Activity

Pipeline
  1. Dataset handling          (load, missing values, duplicates, label encoding)
  2. Dataset analysis          (P(Spam), P(Ham), class-distribution bar chart)
  3. Word-based probability    (P(Spam | word) directly and via Bayes' theorem)
  4. Random variable analysis  (Y ~ Bernoulli(p): E(Y), Var(Y))
  5. Data preparation          (80/20 stratified split, CountVectorizer)
  6. Model building            (Multinomial Naive Bayes)
  7. Model evaluation          (accuracy, precision, recall, F1, confusion matrix)
  8. Real-time testing         (type an SMS, get P(Spam | Message) and P(Ham | Message))

Run:
    python spam_detector.py                 # full pipeline, then interactive prompt
    python spam_detector.py --no-interactive
    python spam_detector.py --predict "Congratulations! You have won a free prize. Claim now."
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files; works on any machine, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "SMSSpamCollection"
OUT_DIR = BASE_DIR / "outputs"
MODEL_FILE = OUT_DIR / "spam_model.joblib"

SUSPICIOUS_WORDS = ["free", "prize", "win", "urgent", "offer"]
RANDOM_STATE = 42


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Dataset handling
# ---------------------------------------------------------------------------
def load_dataset():
    """Load the SMS Spam Collection from the local file; fall back to UCI download."""
    if DATA_FILE.exists():
        df = pd.read_csv(
            DATA_FILE, sep="\t", header=None, names=["label", "message"],
            encoding="utf-8", quoting=3,  # quoting=3 -> QUOTE_NONE (messages contain quotes)
        )
        print(f"Loaded local file: {DATA_FILE.name}")
        return df

    print("Local file not found - downloading from UCI with ucimlrepo ...")
    try:
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=228)
        df = pd.concat([data.data.targets, data.data.features], axis=1)
        df.columns = ["label", "message"]
        return df
    except Exception as exc:
        sys.exit(f"Could not load the dataset ({exc}). Put 'SMSSpamCollection' in the data/ folder.")


def clean_dataset(df):
    heading("1. DATASET HANDLING")
    print(f"Raw shape                : {df.shape}")
    print("Missing values per column:")
    print(df.isnull().sum().to_string())

    df = df.dropna().copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    n_dups = df.duplicated(subset="message").sum()
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    print(f"Duplicate messages removed: {n_dups}")

    df["y"] = df["label"].map({"ham": 0, "spam": 1})  # Spam = 1, Ham = 0
    assert df["y"].notnull().all(), "Unexpected label found"
    df["y"] = df["y"].astype(int)

    print(f"Cleaned shape            : {df.shape}")
    print(df.head().to_string())
    return df


# ---------------------------------------------------------------------------
# 2. Dataset analysis
# ---------------------------------------------------------------------------
def analyse_classes(df):
    heading("2. DATASET ANALYSIS")
    n = len(df)
    n_spam = int((df["y"] == 1).sum())
    n_ham = int((df["y"] == 0).sum())
    p_spam, p_ham = n_spam / n, n_ham / n
    print(f"Total messages : {n}")
    print(f"Spam messages  : {n_spam}")
    print(f"Ham messages   : {n_ham}")
    print(f"P(Spam) = {n_spam}/{n} = {p_spam:.4f}")
    print(f"P(Ham)  = {n_ham}/{n} = {p_ham:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Ham (0)", "Spam (1)"], [n_ham, n_spam], color=["#4C9F70", "#D9534F"])
    for bar, count, prob in zip(bars, [n_ham, n_spam], [p_ham, p_spam]):
        ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count}\n({prob:.1%})",
                ha="center", va="bottom")
    ax.set_ylabel("Number of messages")
    ax.set_title("Class Distribution of SMS Messages")
    ax.set_ylim(0, max(n_ham, n_spam) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved: outputs/class_distribution.png")
    return n_spam, n_ham, p_spam, p_ham


# ---------------------------------------------------------------------------
# 3. Word-based probability analysis
# ---------------------------------------------------------------------------
def contains_word(series, word):
    """True where the message contains `word` as a whole word (case-insensitive, via re)."""
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    return series.apply(lambda text: bool(pattern.search(text)))


def word_probabilities(df, p_spam):
    heading("3. WORD-BASED PROBABILITY ANALYSIS")
    n = len(df)
    is_spam = df["y"] == 1
    rows = []
    for word in SUSPICIOUS_WORDS:
        has_word = contains_word(df["message"], word)
        n_word = int(has_word.sum())
        n_word_spam = int((has_word & is_spam).sum())

        # Direct (counting) definition
        p_direct = n_word_spam / n_word if n_word else float("nan")

        # Bayes' theorem: P(S|W) = P(W|S) * P(S) / P(W)
        p_word = n_word / n
        p_word_given_spam = n_word_spam / is_spam.sum()
        p_bayes = p_word_given_spam * p_spam / p_word if n_word else float("nan")

        rows.append({
            "word": word,
            "msgs_with_word": n_word,
            "spam_with_word": n_word_spam,
            "P(Word)": p_word,
            "P(Word|Spam)": p_word_given_spam,
            "P(Spam|Word) direct": p_direct,
            "P(Spam|Word) Bayes": p_bayes,
            "match": bool(np.isclose(p_direct, p_bayes)),
        })
    table = pd.DataFrame(rows)

    for r in rows:
        w = r["word"]
        print(f"\nWord: '{w}'")
        print(f"  Messages containing '{w}'       : {r['msgs_with_word']}")
        print(f"  Spam messages containing '{w}'  : {r['spam_with_word']}")
        print(f"  Direct : P(Spam|{w}) = {r['spam_with_word']}/{r['msgs_with_word']} = {r['P(Spam|Word) direct']:.4f}")
        print(f"  Bayes  : P({w}|Spam) x P(Spam) / P({w}) = "
              f"{r['P(Word|Spam)']:.4f} x {p_spam:.4f} / {r['P(Word)']:.4f} = {r['P(Spam|Word) Bayes']:.4f}")
        print(f"  Direct and Bayes results agree: {r['match']}")

    table.round(4).to_csv(OUT_DIR / "word_probabilities.csv", index=False)
    print("\nSaved: outputs/word_probabilities.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["word"], table["P(Spam|Word) direct"], color="#D9534F")
    ax.axhline(p_spam, color="black", linestyle="--", label=f"Overall P(Spam) = {p_spam:.3f}")
    for i, v in enumerate(table["P(Spam|Word) direct"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("P(Spam | word)")
    ax.set_title("Probability of Spam Given a Suspicious Word")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "word_probabilities.png", dpi=150)
    plt.close(fig)
    print("Saved: outputs/word_probabilities.png")
    return table


# ---------------------------------------------------------------------------
# 4. Random variable analysis
# ---------------------------------------------------------------------------
def random_variable_analysis(df, p_spam):
    heading("4. RANDOM VARIABLE ANALYSIS")
    print("Y = 1 if the message is spam, Y = 0 if the message is ham  (Bernoulli random variable)")
    e_y = p_spam                      # E(Y) = 1*p + 0*(1-p) = p
    var_y = p_spam * (1 - p_spam)     # Var(Y) = p(1-p)
    print(f"p = P(Spam)        = {p_spam:.4f}")
    print(f"E(Y)   = p         = {e_y:.4f}")
    print(f"Var(Y) = p(1 - p)  = {var_y:.4f}")
    print(f"Std(Y) = sqrt(Var) = {np.sqrt(var_y):.4f}")
    print("Check with numpy on the data:")
    print(f"  np.mean(Y) = {np.mean(df['y']):.4f}   np.var(Y) = {np.var(df['y']):.4f}")
    return e_y, var_y


# ---------------------------------------------------------------------------
# 5-6. Data preparation and model building
# ---------------------------------------------------------------------------
def prepare_and_train(df):
    heading("5. DATA PREPARATION  &  6. MODEL BUILDING")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["message"], df["y"], test_size=0.20, stratify=df["y"], random_state=RANDOM_STATE
    )
    print(f"Training messages: {len(X_train_text)}  (spam share {y_train.mean():.4f})")
    print(f"Testing messages : {len(X_test_text)}  (spam share {y_test.mean():.4f})")

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)   # fit on training data only
    X_test = vectorizer.transform(X_test_text)
    print(f"Vocabulary size  : {len(vectorizer.vocabulary_)} words")
    print(f"Feature matrix   : train {X_train.shape}, test {X_test.shape}")

    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Multinomial Naive Bayes model trained.")

    joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_FILE)
    print(f"Saved: outputs/{MODEL_FILE.name}")
    return model, vectorizer, X_test, X_test_text, y_test


# ---------------------------------------------------------------------------
# 7. Model evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, X_test_text, y_test):
    heading("7. MODEL EVALUATION")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}   (of messages predicted spam, share that really are spam)")
    print(f"Recall    : {rec:.4f}   (of real spam, share that was caught)")
    print(f"F1-score  : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=["Actual Ham", "Actual Spam"],
                       columns=["Pred Ham", "Pred Spam"]).to_string())
    print(f"\nTrue Negatives  (ham -> ham)              : {tn}")
    print(f"True Positives  (spam -> spam)            : {tp}")
    print(f"False Negatives (SPAM classified as HAM)  : {fn}")
    print(f"False Positives (HAM classified as SPAM)  : {fp}")

    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], digits=4)
    print("\nClassification report:\n" + report)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Show the actual misclassified messages
    result = pd.DataFrame({"message": X_test_text.values, "actual": y_test.values, "predicted": y_pred})
    spam_as_ham = result[(result.actual == 1) & (result.predicted == 0)]
    ham_as_spam = result[(result.actual == 0) & (result.predicted == 1)]
    print(f"\nSpam messages incorrectly classified as genuine ({len(spam_as_ham)}):")
    for m in spam_as_ham["message"]:
        print("  -", m[:110])
    print(f"\nGenuine messages incorrectly classified as spam ({len(ham_as_spam)}):")
    for m in ham_as_spam["message"]:
        print("  -", m[:110])

    with open(OUT_DIR / "performance_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}\nF1-score : {f1:.4f}\n\n")
        f.write(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}\n\n")
        f.write(f"Spam classified as ham (FN): {fn}\nHam classified as spam (FP): {fp}\n\n")
        f.write(report)
        f.write("\nSpam incorrectly classified as genuine:\n")
        f.writelines(f"  - {m}\n" for m in spam_as_ham["message"])
        f.write("\nGenuine incorrectly classified as spam:\n")
        f.writelines(f"  - {m}\n" for m in ham_as_spam["message"])
    print("\nSaved: outputs/confusion_matrix.png, outputs/performance_report.txt")


# ---------------------------------------------------------------------------
# 8. Real-time testing
# ---------------------------------------------------------------------------
def predict_message(text, model, vectorizer):
    proba = model.predict_proba(vectorizer.transform([text]))[0]
    p_ham, p_spam = proba[0], proba[1]
    label = "Spam" if p_spam >= 0.5 else "Ham (Genuine)"
    print(f"\nMessage            : {text}")
    print(f"Prediction         : {label}")
    print(f"P(Spam | Message)  = {p_spam:.2f}")
    print(f"P(Ham  | Message)  = {p_ham:.2f}")
    return label, p_spam, p_ham


def interactive_loop(model, vectorizer):
    heading("8. REAL-TIME TESTING")
    print("Type an SMS message and press Enter. Type 'quit' to exit.")
    while True:
        try:
            text = input("\nSMS> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        predict_message(text, model, vectorizer)


def write_summary(df, n_spam, n_ham, p_spam, p_ham, e_y, var_y):
    with open(OUT_DIR / "cleaned_dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("CLEANED DATASET SUMMARY\n=======================\n")
        f.write(f"Total unique messages : {len(df)}\n")
        f.write(f"Spam (Y=1)            : {n_spam}\nHam  (Y=0)            : {n_ham}\n")
        f.write(f"P(Spam)               : {p_spam:.4f}\nP(Ham)                : {p_ham:.4f}\n")
        f.write(f"E(Y)                  : {e_y:.4f}\nVar(Y)                : {var_y:.4f}\n")
        lengths = df["message"].str.len()
        f.write(f"Mean message length   : {lengths.mean():.1f} characters\n")
        f.write(f"Mean length (spam)    : {df.loc[df.y == 1, 'message'].str.len().mean():.1f}\n")
        f.write(f"Mean length (ham)     : {df.loc[df.y == 0, 'message'].str.len().mean():.1f}\n")
    df[["label", "y", "message"]].to_csv(OUT_DIR / "cleaned_dataset.csv", index=False)
    print("Saved: outputs/cleaned_dataset_summary.txt, outputs/cleaned_dataset.csv")


def main():
    parser = argparse.ArgumentParser(description="Probabilistic SMS Spam Detection")
    parser.add_argument("--no-interactive", action="store_true", help="skip the interactive prompt")
    parser.add_argument("--predict", metavar="TEXT", help="classify a single message and exit")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    df = clean_dataset(load_dataset())
    n_spam, n_ham, p_spam, p_ham = analyse_classes(df)
    word_probabilities(df, p_spam)
    e_y, var_y = random_variable_analysis(df, p_spam)
    write_summary(df, n_spam, n_ham, p_spam, p_ham, e_y, var_y)
    model, vectorizer, X_test, X_test_text, y_test = prepare_and_train(df)
    evaluate(model, X_test, X_test_text, y_test)

    if args.predict:
        heading("8. REAL-TIME TESTING")
        predict_message(args.predict, model, vectorizer)
    elif not args.no_interactive:
        interactive_loop(model, vectorizer)


if __name__ == "__main__":
    main()
