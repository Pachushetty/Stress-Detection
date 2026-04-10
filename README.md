# 🧠 Stress Predictor (Flask)
A smart and interactive **Stress Detection web app** built using **Flask, HTML, CSS, and JavaScript**.  
Detect stress levels from natural language input with a clean UI and instant feedback.

---

## 🚀 Features
- 📝 Free-form text input for stress analysis
- 🔍 Hybrid prediction system:
  - ⚡ Keyword-based matching for fast, direct detection
  - 🤖 Naive Bayes ML model as fallback for complex sentences
- 📊 Instant results:
  - ✅ Stress Detected
  - ✅ No Stress Detected
- 🎯 Real-time character-level text preprocessing
  - Stopword removal
  - Snowball stemming
- 🏆 Clear result display with status indicator
- 🔁 Try again with new input instantly

---

## 🛠️ Tech Stack
- **Backend:** Python (Flask), scikit-learn, NLTK
- **Frontend:** HTML, CSS, JavaScript
- **Model:** Bernoulli Naive Bayes
- **Vectoriser:** CountVectorizer

---

## 📁 Project Structure
```
stress_updated/
│
├── stress/
│   ├── app.py               # Flask app — preprocessing, training, and routing
│   ├── stress.csv           # Labelled dataset (text, label)
│   ├── model.pkl            # Serialised trained model
│   ├── cv.pkl               # Serialised CountVectorizer
│   └── templates/
│       └── index.html       # Front-end interface
└── README.md
```

---

## ▶️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Pachushetty/Stress-Detection.git
cd Stress-Detection/stress
```

### 2. Install dependencies
```bash
pip install flask numpy pandas nltk scikit-learn
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000/
```

---

## 🎮 How to Use
1. Open the app in your browser
2. Type a sentence or phrase describing how you feel
3. Click **Analyse** to get your result
4. The app will display whether stress is detected or not
5. Clear the input and try again with a new entry

---

## 🏆 Prediction System

| Method | Trigger | Example |
|---|---|---|
| ⚡ Keyword Match | Single words or short phrases | *"anxious"*, *"burnout"* |
| 🤖 ML Model | Complex or multi-word sentences | *"I have been feeling really off lately"* |

---

## 🔬 How It Works

**Step 1 — Keyword Check**  
The input is scanned against a curated dictionary of stress-related terms (e.g., *anxiety*, *exhausted*, *hopeless*) and positive well-being terms (e.g., *relaxed*, *grateful*, *content*). If a match is found, the result is returned immediately.

**Step 2 — ML Model Fallback**  
If no keyword matches, the text undergoes preprocessing — lowercasing, punctuation removal, stopword filtering, and Snowball stemming — before being vectorised and passed to a trained **Bernoulli Naive Bayes** classifier.

---

## 📌 Future Improvements
- 🌐 Deploy online (Render / Railway)
- 💾 Save prediction history (database)
- 👤 User login system
- 📊 Stress trend dashboard
- 🎨 Light / Dark theme toggle

---

## 👩‍💻 Author
**Prathiksha**
