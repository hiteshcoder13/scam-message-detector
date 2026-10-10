import re
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "scam_model.joblib"

SAFE_LIMIT = 0.30     # below this  -> Safe
SCAM_LIMIT = 0.70     # above this  -> Scam

# Same lists as Step 9 of the training notebook
warning_words = {
    "Asks for private info": ["otp", "pin", "password", "kyc", "cvv", "verify", "aadhaar"],
    "Creates urgency":       ["urgent", "immediately", "act now", "expire", "last chance", "hurry"],
    "Offers a prize":        ["won", "winner", "prize", "lottery", "reward", "claim", "free"],
    "Threatens you":         ["blocked", "suspended", "arrest", "legal action", "police"],
    "Asks for payment":      ["pay", "fee", "transfer", "deposit"],
}

MODEL_NAMES = {"naive_bayes": "Naive Bayes", "logistic": "Logistic Regression", "forest": "Random Forest"}

# Must match clean_text() in the notebook exactly – the model was trained on this output
def clean_text(text):
    text = text.lower()                                          # 1. lowercase
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)       # 2. links
    text = re.sub(r"\d{5,}", " numtoken ", text)                 # 3. long numbers
    text = re.sub(r"[^a-z0-9 ]", " ", text)                      # 4. remove symbols
    text = re.sub(r"\s+", " ", text).strip()                     # remove extra spaces
    return text


def get_verdict(scam_probability):
    if scam_probability < SAFE_LIMIT:
        return "Safe"
    elif scam_probability < SCAM_LIMIT:
        return "Suspicious"
    else:
        return "Scam"


def find_warning_signs(text):
    text = text.lower()
    signs = []
    for sign_name, words in warning_words.items():
        found = [w for w in words if re.search(r"\b" + w + r"\b", text)]
        if found:
            signs.append(sign_name + " → " + ", ".join(found))
    if "http" in text or "www." in text or "bit.ly" in text:
        signs.append("Contains a link")
    return signs


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Scam Message Detector", page_icon="🚨", layout="centered")

st.title("🚨 Scam Message Detector")
st.write(
    "Paste an SMS or chat message and the model will say whether it looks "
    "**✅ Safe**, **⚠️ Suspicious** or **🚨 Scam**."
)

model = load_model()

message = st.text_area("Message", height=140,
                       placeholder="e.g. Your account is blocked. Share the OTP to unblock it.")

if st.button("Check message", type="primary", disabled=not message.strip()):
    cleaned = clean_text(message)
    scam_probability = model.predict_proba([cleaned])[0][1]
    verdict = get_verdict(scam_probability)

    show = {"Safe": st.success, "Suspicious": st.warning, "Scam": st.error}[verdict]
    icon = {"Safe": "✅", "Suspicious": "⚠️", "Scam": "🚨"}[verdict]
    show(f"{icon} **{verdict.upper()}** — scam probability = {scam_probability:.2f}")
    st.progress(float(scam_probability))

    signs = find_warning_signs(message)
    if signs:
        st.subheader("Warning signs found")
        for sign in signs:
            st.write("•", sign)

    with st.expander("How the three models voted"):
        rows = []
        for key, estimator in model.named_estimators_.items():
            p = estimator.predict_proba([cleaned])[0][1]
            rows.append({"Model": MODEL_NAMES.get(key, key), "Scam probability": round(p, 2)})
        rows.append({"Model": "Average (final)", "Scam probability": round(scam_probability, 2)})
        st.table(rows)
        st.caption(f"Safe < {SAFE_LIMIT} ≤ Suspicious < {SCAM_LIMIT} ≤ Scam")
        st.caption(f"Cleaned text sent to the model: `{cleaned}`")

st.divider()
st.caption(
    "Trained on the UCI SMS Spam Collection with Naive Bayes, Logistic Regression and "
    "Random Forest (soft voting). B.Tech CSE mini project, SGT University. "
    "This is a learning project – always double-check suspicious messages yourself."
)
