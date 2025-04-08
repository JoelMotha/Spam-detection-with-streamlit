import streamlit as st
import pickle
import numpy as np

# Load the model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Spam Message Detector")

# Session state to track navigation
if "page" not in st.session_state:
    st.session_state.page = "input"

# Page 1: Input screen
if st.session_state.page == "input":
    st.title("📩 Spam Message Detector")

    message = st.text_area("Enter your message:", height=200)

    if st.button("CHECK SPAM"):
        st.session_state.message = message
        st.session_state.page = "result"

# Page 2: Result screen
elif st.session_state.page == "result":
    st.title("Prediction Result")

    message = st.session_state.get("message", "")
    vec = vectorizer.transform([message])
    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]

    label = "SPAM ⚠️" if pred == 1 else "NOT Spam"
    confidence = np.max(proba)

    st.subheader(f"This message is: **{label}**")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")

    st.markdown("---")
    st.subheader("🛡️ Safety Tips:")
    st.markdown("""
    - **Do not** click unknown links or download attachments from suspicious messages.  
    - **Never** share personal, banking, or login details via text.  
    - Spam often uses urgency, prizes, or fear tactics to trick you. Stay alert!
    """)

    if st.button("Check Another Message"):  
        st.session_state.page = "input"
