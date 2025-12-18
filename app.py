import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- ১. প্রয়োজনীয় ডেটা ডাউনলোড করা (Fixed Version) ---
try:
    # নতুন সংস্করণে 'punkt_tab' এবং 'stopwords' প্রয়োজন হয়
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"NLTK Data Download Error: {e}")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # এক লাইনে ক্লিনিং এবং স্টেমার ব্যবহার
    y = [ps.stem(i) for i in text if i.isalnum() and i not in stop_words and i not in string.punctuation]

    return " ".join(y)

# --- ২. মডেল লোড করা (Relative Path ব্যবহার করুন) ---
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' অথবা 'model.pkl' ফাইলটি খুঁজে পাওয়া যায়নি।")
    st.stop()

# --- ৩. ইউজার ইন্টারফেস ---
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("দয়া করে একটি মেসেজ লিখুন।")
    else:
        # ১. প্রসেসিং
        transformed_sms = transform_text(input_sms)
        # ২. ভেক্টরাইজেশন
        vector_input = tfidf.transform([transformed_sms])
        # ৩. প্রেডিকশন
        try:
            result = model.predict(vector_input)[0]
            # ৪. রেজাল্ট দেখানো
            if result == 1:
                st.header("🚨 Spam")
            else:
                st.header("✅ Not Spam")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("আপনার মডেলটি কি 'fit' করা হয়েছিল? একবার চেক করুন।")
