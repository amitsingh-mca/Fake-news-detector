from flask import Flask, request, render_template
import joblib, os

app = Flask(__name__)
if not os.path.exists("model.joblib"):
    raise RuntimeError("Run train.py first")
model = joblib.load("model.joblib")

@app.route("/", methods=["GET","POST"])
def home():
    prediction=None
    score=None
    text=""
    if request.method=="POST":
        text=request.form.get("news","")
        proba=model.predict_proba([text])[0]
        pred=model.predict([text])[0]
        prediction="Real News" if pred==1 else "Fake News"
        score=round(max(proba)*100,2)
    return render_template("index.html", prediction=prediction, score=score, text=text)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
