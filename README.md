# Probabilistic SMS Spam Detection

BAMAT207 Probability and Statistics - Project Activity (Probability and Random Variables)

A Python system that classifies an SMS as **Spam** or **Ham (genuine)** with a Multinomial Naive Bayes
model and shows the prediction probabilities. Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 labelled messages).

## Project structure

| Path | Purpose |
| --- | --- |
| `spam_detector.py` | Full pipeline as a script, with an interactive real-time tester |
| `SMS_Spam_Detection.ipynb` | Same project as a notebook (outputs already included; also runs on Google Colab) |
| `data/SMSSpamCollection` | Dataset (tab-separated: label, message) |
| `outputs/` | Generated charts, tables, report, cleaned data and saved model |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
python spam_detector.py                # full pipeline, then type your own SMS
python spam_detector.py --no-interactive
python spam_detector.py --predict "Congratulations! You have won a free prize. Claim now."
jupyter notebook SMS_Spam_Detection.ipynb
```

## What it does (maps to the activity sheet)

1. **Dataset handling** - loads data, checks missing values, removes duplicates (414 removed, 5,160 remain), Spam = 1, Ham = 0.
2. **Dataset analysis** - P(Spam), P(Ham), class-distribution bar chart.
3. **Word probabilities** - P(Spam | word) for free, prize, win, urgent, offer, computed by counting and verified with Bayes' theorem.
4. **Random variable** - Y Bernoulli(p): E(Y) = p, Var(Y) = p(1 - p).
5. **Data preparation** - 80/20 stratified split, `CountVectorizer`.
6. **Model** - `MultinomialNB`.
7. **Evaluation** - accuracy, precision, recall, F1, confusion matrix, and the actual misclassified messages.
8. **Real-time testing** - prints Prediction, P(Spam | Message) and P(Ham | Message).

## Results (test set, 1,032 messages)

| Metric | Value |
| --- | --- |
| Accuracy | 0.9826 |
| Precision (spam) | 0.9508 |
| Recall (spam) | 0.9062 |
| F1 (spam) | 0.9280 |
| Spam classified as ham | 12 |
| Ham classified as spam | 6 |

Results are reproducible (`random_state=42`).

## Output files

`class_distribution.png`, `word_probabilities.png` / `.csv`, `confusion_matrix.png`, `performance_report.txt`,
`cleaned_dataset_summary.txt`, `cleaned_dataset.csv`, `spam_model.joblib`

## Upload to GitHub

```bash
git init
git add .
git commit -m "SMS spam detection using Naive Bayes"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
