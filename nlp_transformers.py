from sklearn.datasets import fetch_20newsgroups
import spacy
from transformers import pipeline

# Load dataset
categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
text_sample = newsgroups.data[0]

# spaCy Tokenization & NER
nlp = spacy.load("en_core_web_sm")
doc = nlp(text_sample)

print("Tokens:", [token.text for token in doc[:20]])

print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")

# HuggingFace Transformers Sentiment
classifier = pipeline("sentiment-analysis")
print("\nSentiment:", classifier(text_sample[:500]))