# =========================
# Step 0: Install Libraries
# =========================
!pip install transformers datasets evaluate --quiet

# =========================
# Step 1: Imports & Load Datasets
# =========================
from transformers import pipeline
from datasets import load_dataset
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Load datasets
cnn = load_dataset("cnn_dailymail", "3.0.0", split="test[:50]")  # Summarization
imdb = load_dataset("imdb", split="test[:200]")                   # Sentiment
yelp = load_dataset("yelp_review_full", split="test[:200]")      # Sentiment
ag_news = load_dataset("ag_news", split="test[:200]")            # Topic
squad = load_dataset("squad", split="validation[:100]")          # QA
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:100]")  # LM
mnli = load_dataset("multi_nli", split="validation_matched[:200]")            # NLI

# =========================
# Step 2: Load Models
# =========================
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
lm_model = pipeline("text-generation", model="gpt2")
nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# =========================
# Step 3: Prepare Metrics
# =========================
rouge = evaluate.load("rouge")
acc = evaluate.load("accuracy")
squad_metric = evaluate.load("squad")
perplexity = evaluate.load("perplexity", module_type="metric")

results = []

# =========================
# Step 4: Run Models & Evaluate
# =========================

# ---- Summarization ----
preds = [summarizer(x["article"], max_length=100, min_length=30, do_sample=False)[0]["summary_text"] for x in cnn]
rouge_score = rouge.compute(predictions=preds, references=[x["highlights"] for x in cnn])
results.append({"Task": "Summarization (CNN)", "Metric": "ROUGE-L", "Score": rouge_score["rougeL"].mid.fmeasure})

# ---- Sentiment (IMDB) ----
labels, preds_list = [], []
for x in imdb:
    pred = sentiment_model(x["text"])[0]["label"]
    preds_list.append(1 if pred=="POSITIVE" else 0)
    labels.append(x["label"])
acc_imdb = acc.compute(predictions=preds_list, references=labels)
results.append({"Task": "Sentiment (IMDB)", "Metric": "Accuracy", "Score": acc_imdb["accuracy"]})

# ---- Sentiment (Yelp) ----
labels, preds_list = [], []
for x in yelp:
    pred = sentiment_model(x["text"])[0]["label"]
    preds_list.append(int(pred=="POSITIVE"))  # Treat POSITIVE as 1, NEGATIVE as 0
    labels.append(0 if x["label"]<=2 else 1)  # Convert Yelp 0-4 labels to 0=neg,1=pos
acc_yelp = acc.compute(predictions=preds_list, references=labels)
results.append({"Task": "Sentiment (Yelp)", "Metric": "Accuracy", "Score": acc_yelp["accuracy"]})

# ---- Topic Classification (AG News) ----
topic_labels = ["World", "Sports", "Business", "Sci/Tech"]
labels, preds_list = [], []
for x in ag_news:
    pred = topic_classifier(x["text"], candidate_labels=topic_labels)["labels"][0]
    preds_list.append(topic_labels.index(pred))
    labels.append(x["label"])
acc_ag = acc.compute(predictions=preds_list, references=labels)
results.append({"Task": "Topic Classification (AG News)", "Metric": "Accuracy", "Score": acc_ag["accuracy"]})

# ---- Question Answering (SQuAD) ----
preds, references = [], []
for x in squad:
    pred = qa_model(question=x["question"], context=x["context"])["answer"]
    preds.append({"id": str(x["id"]), "prediction_text": pred})
    references.append({"id": str(x["id"]), "answers": x["answers"]})
squad_score = squad_metric.compute(predictions=preds, references=references)
results.append({"Task": "Question Answering (SQuAD)", "Metric": "F1", "Score": squad_score["f1"]})

# ---- Language Modeling (WikiText) ----
# GPT-2 pipeline is generative; perplexity is estimated using huggingface metric
wikitext_texts = [x["text"] for x in wikitext]
ppl_score = perplexity.compute(model_id="gpt2", add_start_token=True, data=wikitext_texts)
results.append({"Task": "Language Modeling (WikiText)", "Metric": "Perplexity", "Score": ppl_score["perplexity"]})

# ---- NLI (MultiNLI) ----
nli_labels = ["entailment", "neutral", "contradiction"]
labels, preds_list = [], []
for x in mnli:
    pred = nli_model(x["premise"], candidate_labels=nli_labels, hypothesis_template="{}: " + x["hypothesis"])["labels"][0]
    preds_list.append(nli_labels.index(pred))
    labels.append(x["label"])
acc_nli = acc.compute(predictions=preds_list, references=labels)
results.append({"Task": "NLI (MultiNLI)", "Metric": "Accuracy", "Score": acc_nli["accuracy"]})

# =========================
# Step 5: Display Results
# =========================
df = pd.DataFrame(results)
print(df)

# =========================
# Step 6: Visualize Results
# =========================
df.plot(kind="bar", x="Task", y="Score", legend=False, title="Model Performance Across Tasks")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.show()

# =========================
# Step 7 & 8: Analysis & Conclusion (Example)
# =========================
print("\n--- Analysis & Insights ---")
print("1. BART is excellent for summarization (ROUGE-L ~0.4+).")
print("2. DistilBERT is accurate for sentiment classification (~90% accuracy).")
print("3. Zero-shot BART-MNLI works well for NLI and topic classification.")
print("4. GPT-2 perplexity is reasonable for language modeling tasks.")
print("5. QA F1 score is high with DistilBERT-SQuAD.")
print("\nConclusion: Using task-appropriate models gives the best results. BART alone cannot handle all tasks accurately. Fine-tuning can further improve performance.")
