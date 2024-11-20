import streamlit as st
import re
import string
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Function to dynamically apply styles
def add_dynamic_styles():
    st.markdown("""
        <style>
        /* General App Background */
        .stApp {
            background: linear-gradient(to bottom, white, white),
                        url("https://images.unsplash.com/photo-1501785888041-af3ef285b470?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") no-repeat center center fixed;
            background-size: cover;
        }
        [data-theme="dark"] .stApp {
            background: linear-gradient(to bottom, #121212, #121212),
                        url("https://images.unsplash.com/photo-1501785888041-af3ef285b470?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") no-repeat center center fixed;
            background-size: cover;
        }

        /* Header Styling */
        .header-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: black !important;
            margin-bottom: 1.5rem;
        }
        [data-theme="dark"] .header-title {
            color: white !important;
        }

        /* Input Box Styling */
        .stTextInput > div > div {
            background-color: white !important;
            color: black !important;
            border: 2px solid #ccc !important;
            border-radius: 8px !important;
        }
        [data-theme="dark"] .stTextInput > div > div {
            background-color: #333333 !important;
            color: white !important;
            border: 2px solid #555555 !important;
        }

        /* Placeholder Text Styling */
        .stTextInput input::placeholder {
            color: gray !important;
        }
        [data-theme="dark"] .stTextInput input::placeholder {
            color: #aaaaaa !important;
        }

        /* Button Styling */
        .stButton button {
            background-color: #FF8C00 !important;
            color: white !important;
            border-radius: 8px !important;
            font-size: 1.2rem !important;
            font-weight: bold !important;
        }
        .stButton button:hover {
            background-color: #FFA500 !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Header Function
def add_header():
    st.markdown("""
        <div>
            <h1 class="header-title">🌍 Safari Travel Advisor</h1>
        </div>
    """, unsafe_allow_html=True)

# Load preprocessed dataset dynamically
@st.cache_data
def load_preprocessed_data():
    return pd.read_csv('/mount/src/travel-wordfinder/deployment/preprocessed_df.csv')

preprocessed_df = load_preprocessed_data()

# Preprocess text using NLTK lemmatizer
def preprocess_text(text):
    """
    Preprocess raw text input.
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words containing numbers
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return [" ".join(lemmatized_tokens)]

# Load pre-trained model and vectorizer
final_model = joblib.load('/mount/src/travel-wordfinder/deployment/final_model.pkl')
vectorizer_final = joblib.load('/mount/src/travel-wordfinder/deployment/vectorizer_final.pkl')

# Apply styles and header
add_dynamic_styles()
add_header()

# User Input and Prediction
st.markdown("""
    <div style="text-align: right; margin-bottom: 1rem; font-size: 1.2rem;">
        Enter your dream travel experience below, and let us find the perfect destination for you!
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_query = st.text_input(
        "**What's your ideal travel experience?**",
        placeholder="E.g., 'Alpine meadows and glaciers, wildlife viewing...'"
    ).strip()
    predict_button = st.button("🔍 Find My Perfect Destination")

if predict_button:
    if user_query:
        try:
            processed_input = preprocess_text(user_query)
            predicted_country = final_model.predict(vectorizer_final.transform(processed_input))[0]
            st.markdown(f"""
                <div style="text-align: center; margin-top: 2rem;">
                    <h2>Your Perfect Destination:</h2>
                    <h1 style="color: #FF8C00;">{predicted_country} 🎯</h1>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("An error occurred while processing your request.")
            st.exception(e)
    else:
        st.warning("Please enter your travel preferences to get recommendations!")

st.markdown("""
    <div class="footer">
        <p>View the code on <a href="https://github.com/KImondorose/Travel-WordFinder" target="_blank">GitHub</a>.</p>
    </div>
""", unsafe_allow_html=True)
