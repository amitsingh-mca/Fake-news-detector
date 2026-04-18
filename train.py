import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = [
("Government announces new policy for education reform",1),
("Scientists discover promising cancer treatment in trials",1),
("Local city opens new public library downtown",1),
("Aliens secretly control world leaders says leaked source",0),
("Drinking bleach cures every disease instantly",0),
("Celebrity is 300 years old claims anonymous blog",0),
("Election commission releases official voting schedule",1),
("Hidden chip in rain controls your thoughts",0),
]
df=pd.DataFrame(data, columns=["text","label"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=500))
])
model.fit(df["text"], df["label"])
joblib.dump(model, "model.joblib")
print("Model trained and saved.")
