from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Create folders if they don't exist
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]

    if file.filename == "":
        return "No file selected."

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    # Read CSV
    df = pd.read_csv(filepath)

    if "review" not in df.columns:
        return "CSV must contain a column named 'review'."

    # Convert reviews into TF-IDF features
    X = vectorizer.transform(df["review"])

    # Predict
    predictions = model.predict(X)

    # Convert prediction values
    df["Prediction"] = [
        "Positive" if p == 1 else "Negative"
        for p in predictions
    ]

    output_path = os.path.join(
        RESULT_FOLDER,
        "prediction_results.csv"
    )

    df.to_csv(output_path, index=False)

    return send_file(
        output_path,
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)
