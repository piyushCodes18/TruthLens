
import joblib

model = joblib.load("fake_news_welfake_model.pkl")
vectorizer = joblib.load("welfake_tfidf_vectorizer.pkl")

headline = input("Enter news headline: ")
article = input("Enter article text: ")

text = headline + " " + article
features = vectorizer.transform([text])

prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

classes = list(model.classes_)
real_index = classes.index(0)
fake_index = classes.index(1)

fake_percent = probabilities[fake_index] * 100
real_percent = probabilities[real_index] * 100

print("\n--- TruthLens WELFake Model Result ---")
print(f"FAKE: {fake_percent:.2f}%")
print(f"REAL: {real_percent:.2f}%")
print("Prediction:", "REAL" if prediction == 0 else "FAKE")
