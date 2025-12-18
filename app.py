import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- 1. Setup & Resource Downloading ---
# This ensures the app doesn't crash on servers like Streamlit Cloud
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- 2. Preprocessing Function ---
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize into words
    text = nltk.word_tokenize(text)

    # Remove special characters, stopwords, and punctuation, then Stem
    cleaned_text = [
        ps.stem(word) 
        for word in text 
        if word.isalnum() and word not in stop_words and word not in string.punctuation
    ]

    return " ".join(cleaned_text)

# --- 3. Load Saved Pickles ---
# Use relative paths. Ensure these files are in the same folder as app.py
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' or 'model.pkl' not found in the current directory.")
    st.stop()

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("Email/SMS Spam Classifier")
st.markdown("This app uses a Machine Learning model to predict if a message is Spam or Ham.")

input_sms = st.text_area("Enter the message below:", height=150)

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        # We wrap this in a try-except to catch the NotFittedError specifically
        try:
            result = model.predict(vector_input)[0]
            
            # 4. Display Result
            if result == 1:
                st.error("### 🚨 This is Spam")
            else:
                st.success("### ✅ This is Not Spam (Ham)")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: This often happens if the model was not trained/fitted before being pickled.")

# --- 5. Footer ---
st.sidebar.info("How to run: \n1. Open terminal \n2. Type: streamlit run app.py")
# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open(
#     r'C:\Users\USER\OneDrive\Desktop\Code practice\Machine_Learning_Projects\Email Spam Classifier\vectorizer.pkl',
#     'rb'
# ))

# model = pickle.load(open(
#     r'C:\Users\USER\OneDrive\Desktop\Code practice\Machine_Learning_Projects\Email Spam Classifier\model.pkl',
#     'rb'
# ))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")


# # cd "Machine_Learning_Projects/Email Spam Classifier"
# # streamlit run app.py