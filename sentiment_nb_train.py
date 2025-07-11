import pandas as pd
import string
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# === Ganti nama file CSV di sini ===
df = pd.read_csv('sentiment.csv')  # pastikan file csv kamu ada di folder yang sama

# Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

df['clean_text'] = df['clean_text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentimen'], test_size=0.2, random_state=42
)

# TF-IDF + Naive Bayes pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, 'naive_bayes_sentiment_model.pkl')
print("Model berhasil disimpan!")
