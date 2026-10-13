from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
import requests

app = Flask(__name__)

model = joblib.load("fake_news_welfake_model.pkl")
vectorizer = joblib.load("welfake_tfidf_vectorizer.pkl")

# In the supplied CSV, label 0 examples include Reuters-style real news,
# while label 1 examples include sensational fake-news articles.
# This app therefore maps model class 0 to REAL and class 1 to FAKE.
REAL_CLASS = 0
FAKE_CLASS = 1


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    article_text = str(data.get("text", "")).strip()

    if not article_text:
        return jsonify({"error": "Please enter some news."}), 400

    features = vectorizer.transform([article_text])
    probabilities = model.predict_proba(features)[0]
    print("Raw model prediction:", model.predict(features)[0])
    print("Model classes:", model.classes_)
    print("Probabilities:", probabilities)
    class_probabilities = dict(zip(model.classes_, probabilities))

    real_percentage = round(float(class_probabilities.get(REAL_CLASS, 0.0)) * 100, 2)
    fake_percentage = round(float(class_probabilities.get(FAKE_CLASS, 0.0)) * 100, 2)

    result = "REAL" if real_percentage >= fake_percentage else "FAKE"
    percentage = max(real_percentage, fake_percentage)

    fact_check_status = "not_configured"
    fact_check_results = []
    api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

    if api_key:
        try:
            query = re.sub(r"\s+", " ", article_text).strip()[:500]
            response = requests.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={
                    "query": query,
                    "key": api_key,
                    "pageSize": 5,
                    "languageCode": "en",
                },
                timeout=8,
            )
            if response.status_code == 200:
                fact_check_status = "success"
                for claim in response.json().get("claims", []):
                    for review in claim.get("claimReview", []):
                        fact_check_results.append({
                            "claim": claim.get("text", ""),
                            "publisher": review.get("publisher", {}).get("name", "Unknown"),
                            "rating": review.get("textualRating", "Not provided"),
                            "url": review.get("url", ""),
                        })
            else:
                fact_check_status = f"api_error_{response.status_code}"
        except requests.RequestException:
            fact_check_status = "request_failed"

    return jsonify({
        "prediction": result,
        "percentage": percentage,
        "real_percentage": real_percentage,
        "fake_percentage": fake_percentage,
        "fact_check_status": fact_check_status,
        "fact_check_results": fact_check_results,
    })


if __name__ == "__main__":
    app.run(debug=True)
