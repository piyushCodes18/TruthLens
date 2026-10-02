from flask import Flask, request, jsonify, render_template
import joblib
import re
import string

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("bow_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Please enter some news."}), 400

    cleaned_text = clean_text(text)

    news_bow = vectorizer.transform([cleaned_text])

    prediction = model.predict(news_bow)[0]

    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)