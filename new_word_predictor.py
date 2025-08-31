import streamlit as st
import random
import re
from collections import defaultdict
from transformers import pipeline

# -----------------------------
# Part 1: Build N-gram Model
# -----------------------------
def build_ngram_model(corpus, n=3):
    model = defaultdict(list)
    words = re.findall(r"\w+", corpus.lower())
    for i in range(len(words) - n):
        prefix = tuple(words[i:i + n - 1])
        next_word = words[i + n - 1]
        model[prefix].append(next_word)
    return model

def predict_ngram(model, text, n=3, k=3):
    words = re.findall(r"\w+", text.lower())
    prefix = tuple(words[-(n - 1):]) if len(words) >= (n - 1) else tuple(words)
    if prefix in model:
        return random.sample(model[prefix], min(k, len(model[prefix])))
    else:
        return ["(no prediction)"]

# -----------------------------
# Larger sample corpus
# -----------------------------
sample_corpus = """
I am going to school. I am going home. I like to play football. 
He is going to market. She is playing with friends. 
This is a simple test corpus for ngram model.
Learning Python is fun. Streamlit makes building apps easy.
Machine learning can predict text. Natural language processing is powerful.
I love to read books. I enjoy coding in Python. Data science is amazing.
"""
ngram_model = build_ngram_model(sample_corpus, n=3)

# -----------------------------
# Part 2: Transformer Model
# -----------------------------
@st.cache_resource
def load_transformer():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=-1)

generator = load_transformer()

def predict_transformer(text, k=3):
    output = generator(text, max_length=len(text.split()) + 5, num_return_sequences=1)
    generated = output[0]['generated_text']
    next_words = generated[len(text):].strip().split()
    return next_words[:k] if next_words else ["(no prediction)"]

# -----------------------------
# Part 3: Streamlit UI
# -----------------------------
st.title("🔮 Hybrid Next-Word Predictor (N-gram + Transformer)")

# User input
user_input = st.text_input("Type your text here:")

# Model choice
model_choice = st.radio("Choose prediction model:", ["N-gram (basic)", "Transformer (advanced)"])

if user_input:
    if model_choice == "N-gram (basic)":
        predictions = predict_ngram(ngram_model, user_input, n=3)
    else:
        predictions = predict_transformer(user_input, k=3)

    # Join list into readable string
    st.write("👉 Suggestions:", ", ".join(predictions))
