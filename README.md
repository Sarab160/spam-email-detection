# 🚀 Spam Email Detector

A **Streamlit web app** that detects whether an email is **Spam** or **Not Spam** using a **Multinomial Naive Bayes** model trained on email text data.

---

## 📌 Features

- Predicts spam emails in real-time by analyzing email content.
- Cleans and preprocesses text (removes URLs, special characters, converts to lowercase).
- Uses TF-IDF Vectorizer for feature extraction.
- Built with Streamlit for an interactive and user-friendly interface.

---

## 📂 Dataset

- CSV file: `emails.csv`
- Columns:
  - `text` → Email content
  - `spam` → Label (1 = Spam, 0 = Not Spam)

---

## 🔧 Tech Stack

- **Python 3.9+**
- **Streamlit** — web app interface
- **Pandas** — data handling
- **Scikit-learn** — ML model and text preprocessing
- **Regex** — text cleaning

-
   git clone <repo-url>
   cd <repo-folder>
