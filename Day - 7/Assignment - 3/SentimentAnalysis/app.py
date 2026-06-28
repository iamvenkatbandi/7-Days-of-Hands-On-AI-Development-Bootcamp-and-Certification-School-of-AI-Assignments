from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from scipy.special import expit

app = Flask(__name__)

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    df = pd.read_csv(filepath)

    X = vectorizer.transform(df["review"])

    predictions = model.predict(X)

    # Confidence Scores
    decision_scores = model.decision_function(X)

    confidence = expit(abs(decision_scores)) * 100

    df["Prediction"] = [
        "Positive" if p == 1 else "Negative"
        for p in predictions
    ]

    df["Confidence (%)"] = confidence.round(2)

    output_path = os.path.join(
        RESULT_FOLDER,
        "prediction_results.csv"
    )

    df.to_csv(output_path, index=False)

    # Count predictions
    positive = sum(predictions == 1)
    negative = sum(predictions == 0)

    # Create Bar Chart
    plt.figure(figsize=(6,4))

    plt.bar(
        ["Positive","Negative"],
        [positive,negative]
    )

    plt.title("Prediction Distribution")

    plt.xlabel("Sentiment")

    plt.ylabel("Count")

    chart_path = os.path.join(
        STATIC_FOLDER,
        "prediction_chart.png"
    )

    plt.savefig(chart_path)

    plt.close()

    table = df.to_html(
        classes="table table-striped",
        index=False
    )

    return render_template(
        "result.html",
        table=table,
        positive=positive,
        negative=negative
    )


if __name__ == "__main__":
    app.run(debug=True)