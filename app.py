from flask import Flask, render_template, request
import joblib
import re
import os

# Base folder (where app.py and index.html live)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tell Flask templates are in the same folder as app.py
app = Flask(__name__, template_folder=".")

# Load saved model and vectorizer
model = joblib.load(os.path.join(BASE_DIR, "resume_classifier.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# Text cleaning function (same as in Colab)
def clean_resume(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        resume_text = request.form.get("resume_text", "")
        clean_text = clean_resume(resume_text)
        X = vectorizer.transform([clean_text])
        prediction = model.predict(X)[0]

    # Renders your big HTML above and injects {{ prediction }}
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    # Render sets PORT env variable; default 5000 for local runs
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
