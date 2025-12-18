import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    cleaned_text = [
        ps.stem(word) 
        for word in text 
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(cleaned_text)


try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' or 'model.pkl' not found in the current directory.")
    st.stop()


st.set_page_config(page_title="Spam Classifier")
st.title("Email/SMS Spam Classifier")
st.markdown("This app uses a Machine Learning model to predict if a message is Spam or Ham.")

input_sms = st.text_area("Enter the message below:", height=150)

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        transformed_sms = transform_text(input_sms)
        st.markdown("**Transformed Text:**")
        st.write(transformed_sms)

        vector_input = tfidf.transform([transformed_sms])

        try:
            result = model.predict(vector_input)[0]
            if result == 1:
                st.error("### 🚨 This is Spam")
            else:
                st.success("### ✅ This is Not Spam (Ham)")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: This often happens if the model was not trained/fitted before being pickled.")
