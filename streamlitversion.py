# spam_email_app.py

import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ------------------ Load Dataset ------------------
df = pd.read_csv("emails.csv")  # Make sure your CSV has 'text' and 'spam' columns

# ------------------ Text Cleaning Function ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)       
    text = re.sub(r'[^a-z\s]', '', text)      
    return text

df['text'] = df['text'].apply(clean_text)

# ------------------ Train Model ------------------
X = df['text']
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(stop_words='english', min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ------------------ Streamlit UI ------------------
st.title("Spam Email Detector 🚀")
st.write("Enter the email content below to check if it is Spam or Not Spam.")

# User Input
user_input = st.text_area("Email Content", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter email content!")
    else:
        # Preprocess input
        clean_input = clean_text(user_input)
        input_tfidf = tfidf.transform([clean_input])
        
        # Prediction
        pred = model.predict(input_tfidf)[0]
        result = "Spam 🚫" if pred == 1 else "Not Spam ✅"
        
        st.success(f"Prediction: {result}")
