import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam Email Classifier")
st.write("Enter a message and check whether it is Spam or Not Spam (Ham).")

text = st.text_area("Enter message/email text:")

if st.button("Check"):
    if text.strip() == "":
        st.warning("Please enter a message.")
    else:
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]

        if prediction == 1:
            st.error("SPAM ❌")
        else:
            st.success("NOT SPAM (HAM) ✅")
