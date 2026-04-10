from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("stress.csv")

# Preprocessing
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# ----------------------------------------------------------------
# Stress keyword dictionary for direct keyword matching
# ----------------------------------------------------------------
STRESS_KEYWORDS = [
    "headache", "migraine", "anxiety", "panic", "overwhelmed", "exhausted",
    "burnout", "depressed", "depression", "hopeless", "crying", "insomnia",
    "worried", "fear", "scared", "nervous", "stressed", "stress", "tension",
    "pressure", "frustration", "frustrated", "angry", "anger", "irritated",
    "irritable", "lonely", "loneliness", "helpless", "worthless", "nausea",
    "palpitations", "sweating", "trembling", "shaking", "dizzy", "dizziness",
    "fatigue", "fatigued", "tired", "pain", "ache", "sick", "unwell",
    "restless", "worry", "overthinking", "overthink", "suffering", "trauma",
    "grief", "abuse", "nightmare", "suicidal", "breakdown", "struggling",
    "dread", "doom", "failure", "failing", "hopelessness", "burdened",
    "fever", "ill", "phobia", "anxious", "distressed", "miserable",
    "unhappy", "sad", "sorrow", "sorrowful", "dejected", "despondent",
    "gloomy", "melancholy", "heartbroken", "disturbed", "troubled",
    "agitated", "tense", "uneasy", "uncomfortable", "sleepless",
    "drained", "weak", "lightheaded", "chest pain", "racing heart",
    "muscle tension", "back pain", "neck pain", "stomach ache",
    "appetite loss", "no motivation", "lost interest", "dark thoughts",
    "feel empty", "feel numb", "feel broken", "not okay", "hate my life",
    "everything hurts", "chronic pain", "health anxiety", "feel sick",
    "feel terrible", "feel awful", "feel horrible", "feel sad", "feel down",
    "feel low", "bad mood", "mood swings", "emotional", "hard time",
    "difficult time", "burnedout", "burnt out"
]

NO_STRESS_KEYWORDS = [
    "happy", "joyful", "excited", "relaxed", "calm", "peaceful", "motivated",
    "energetic", "refreshed", "grateful", "thankful", "blessed", "content",
    "satisfied", "fulfilled", "accomplished", "proud", "confident", "optimistic",
    "hopeful", "positive", "wonderful", "fantastic", "amazing", "great",
    "cheerful", "vibrant", "lighthearted", "balanced", "comfortable", "secure",
    "loved", "supported", "healthy", "well rested", "good sleep", "feeling good",
    "feeling great", "feeling fine", "doing well", "all good", "no worries",
    "stress free", "worry free", "at peace", "at ease", "feeling safe"
]


def keyword_predict(text):
    """
    Fast keyword-based stress check.
    Returns 'Stress', 'No Stress', or None (defer to ML model).
    """
    lower = text.lower()
    for kw in STRESS_KEYWORDS:
        if kw in lower:
            return "Stress"
    for kw in NO_STRESS_KEYWORDS:
        if kw in lower:
            return "No Stress"
    return None  # let ML decide


def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


# Keep only text and label columns
data = data[["text", "label"]].copy()
data["text"] = data["text"].apply(clean)
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})

# Train ML model
cv = CountVectorizer()
X = cv.fit_transform(data["text"])

with open('cv.pkl', 'wb') as f:
    pickle.dump(cv, f)

xtrain, xtest, ytrain, ytest = train_test_split(
    X, data["label"], test_size=0.33, random_state=42
)

model = BernoulliNB()
model.fit(xtrain, ytrain)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['message'].strip()

        if not user_input:
            return render_template('index.html',
                                   prediction_text='Please enter some text.')

        # Step 1: Keyword check (accurate for single words/short phrases)
        kw_result = keyword_predict(user_input)
        if kw_result is not None:
            prediction = kw_result
        else:
            # Step 2: ML model fallback for longer/complex sentences
            data_input = cv.transform([user_input]).toarray()
            prediction = model.predict(data_input)[0]

        if prediction == "Stress":
            icon = "WARNING"
            label = "Stress Detected"
        else:
            icon = "OK"
            label = "No Stress Detected"

        result_text = '{} - Predicted Stress Level: {}'.format(icon, label)
        return render_template('index.html',
                               prediction_text=result_text,
                               is_stress=(prediction == "Stress"))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
