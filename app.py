import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# NLTK Setup
# -----------------------------
ps = PorterStemmer()

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

# -----------------------------
# Text Transformation Function
# -----------------------------
def transform_text(text):
    # Ensure punkt tokenizer is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    text = text.lower()
    text = nltk.word_tokenize(text)

    cleaned_text = [
        ps.stem(word)
        for word in text
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(cleaned_text)

# -----------------------------
# Load Vectorizer and Model
# -----------------------------
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' or 'model.pkl' not found in the current directory.")
    st.stop()

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Spam Classifier")
st.title("Email/SMS Spam Classifier")
st.markdown("This app uses a Machine Learning model to predict if a message is Spam or Ham.")

# User input
input_sms = st.text_area("Enter the message below:", height=150)

# Predict button
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform text
        transformed_sms = transform_text(input_sms)
        st.markdown("**Transformed Text:**")
        st.write(transformed_sms)

        # Vectorize and predict
        vector_input = tfidf.transform([transformed_sms])
        try:
            result = model.predict(vector_input)[0]

            if result == 1:
                st.error("### 🚨 This is Spam")
            else:
                st.success("### ✅ This is Not Spam (Ham)")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Ensure the model was trained and pickled correctly.")
