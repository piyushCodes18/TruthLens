
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("WELFake_Dataset.csv")

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.dropna(subset=["title", "text", "label"])

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

df = df[df["label"].isin([0, 1])]

df["content"] = (
    df["title"].astype(str) + " " + df["text"].astype(str)
).str.strip()

df = df.drop_duplicates(subset=["content"])
df = df[df["content"].str.len() > 20]

X = df["content"]
y = df["label"]

print("Dataset size:", len(df))
print("\nLabel distribution:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    min_df=3,
    max_df=0.98,
    sublinear_tf=True,
    stop_words="english",
    dtype="float32"
)


X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vectorized, y_train)

predictions = model.predict(X_test_vectorized)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["REAL", "FAKE"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "fake_news_welfake_model.pkl")
joblib.dump(vectorizer, "welfake_tfidf_vectorizer.pkl")

print("\nNew model saved successfully.")
print("fake_news_welfake_model.pkl")
print("welfake_tfidf_vectorizer.pkl")
