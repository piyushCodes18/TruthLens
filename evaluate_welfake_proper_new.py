import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET = "WELFake_Dataset.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET)
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

print("Dataset size after cleaning:", len(df))
print("Label distribution:")
print(y.value_counts().sort_index().rename(index={0: "REAL (0)", 1: "FAKE (1)"}))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining a fresh evaluation model on the training split...")
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
nasa_text = """NASA’s SpaceX Crew-12 Splashes Down After 237 Days in Space

NASA’s SpaceX Crew-12 mission safely returned four astronauts to Earth on October 8, 2026, after more than seven months aboard the International Space Station. The SpaceX Dragon spacecraft splashed down in the Pacific Ocean off the coast of Los Angeles, California.

The crew included NASA astronauts Jessica Meir and Jack Hathaway, European Space Agency astronaut Sophie Adenot, and Russian cosmonaut Andrey Fedyaev. The astronauts spent 237 days living and working in orbit, travelling more than 100 million miles and completing over 3,792 orbits around Earth.

During the mission, the crew conducted scientific experiments involving stem cell production, medical research, nutrition, and the effects of spaceflight on the human body. These activities help scientists develop technologies and knowledge needed for future human exploration of the Moon and Mars.

After splashdown, recovery teams retrieved the spacecraft and astronauts for standard post-flight medical checks. NASA described the mission as an example of how international cooperation and commercial spacecraft support scientific research aboard the International Space Station."""

nasa_headline = "NASA’s SpaceX Crew-12 Splashes Down After 237 Days in Space"

nasa_body = nasa_text.replace(nasa_headline, "").strip()

for name, text in [
    ("Headline only", nasa_headline),
    ("Article body only", nasa_body),
    ("Full article", nasa_text)
]:
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    print(f"\n--- {name} ---")
    print("Prediction:", "REAL" if prediction == 0 else "FAKE")
    print("Probabilities [REAL, FAKE]:", probabilities)
predictions = model.predict(X_test_vectorized)

print("\n--- Held-Out Evaluation (unseen test split) ---")
print("Training articles:", len(X_train))
print("Test articles:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["REAL (label 0)", "FAKE (label 1)"],
    zero_division=0
))
print("Confusion matrix (rows = actual, columns = predicted; order: REAL, FAKE):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nThis script does not overwrite your saved model or vectorizer.")
