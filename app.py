import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    cleaned_text = [
        ps.stem(word) 
        for word in text 
        if word.isalnum() and word not in stop_words and word not in string.punctuation
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



# # cd "Machine_Learning_Projects/Email Spam Classifier"
# # streamlit run app.py


# Spam Messages

# Congratulations! You have won a $1000 Walmart Gift Card.
# Click here to claim now  http://fake-link.com

# URGENT! Your account has been compromised.
# Verify immediately to avoid suspension.

# Win FREE cash prizes today!!!
# Text WIN to 90999 now!

# You are selected for a limited-time offer.
# Get cheap loans with 0% interest. Apply now!

#  You’ve won a free iPhone 15!
# Click the link and confirm your details.

# Dear customer, your ATM card will be blocked.
# Call this number immediately.

# Earn $500 per day working from home.
# No experience needed. Register now!


#  Ham / Not Spam Messages 

# Hey, are we still meeting at 5 PM today?

# Don’t forget to submit the assignment before midnight.

# I’ll call you after my class finishes.

# Happy Birthday 
# Hope you have a great day!

# The exam has been postponed to next Monday.

# Can you send me the notes from yesterday’s lecture?

# Let’s watch a movie tonight 




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