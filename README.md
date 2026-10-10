# 🚨 Scam Message Detector

A Streamlit web app that labels a text message as **✅ Safe**, **⚠️ Suspicious** or **🚨 Scam**.

B.Tech CSE AI mini project, SGT University. Team: Babita (261302060), Yashasvi (261359010), Ankita (261302091).

## How it works
- **Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 SMS)
- **Features:** TF-IDF (single words and word pairs)
- **Models:** Naive Bayes, Logistic Regression and Random Forest, combined by soft voting (their scam probabilities are averaged)
- **Verdict:** probability < 0.30 → Safe, 0.30–0.70 → Suspicious, ≥ 0.70 → Scam
- **Test set results:** 99% accuracy, 0 of 725 safe messages flagged as scam, 3 of 112 scams missed

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `scam_model.joblib` | Trained model (created by the notebook) |
| `Scam_Message_Detector_Training (2).ipynb` | Training notebook |
| `requirements.txt` | Packages for the app (used by Streamlit Community Cloud) |
| `requirements-train.txt` | Extra packages for running the notebook |
| `test_messages.txt` | Sample messages to try in the app, with expected results |

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io → **Create app** → choose the repo and branch, and set **Main file path** to `app.py`.
3. Under **Advanced settings**, choose **Python 3.12**.
4. Click **Deploy**.

> If you retrain the model with a different scikit-learn version, update the version in `requirements.txt` to match.
