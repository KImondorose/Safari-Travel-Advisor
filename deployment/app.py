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
        :root {
            --background-light: white;
            --text-color-light: black;
            --input-bg-light: white;
            --input-text-light: black;
            --placeholder-light: gray;

            --background-dark: #121212;
            --text-color-dark: white;
            --input-bg-dark: #333333;
            --input-text-dark: white;
            --placeholder-dark: #aaaaaa;
        }

        /* General App Background */
        .stApp {
            background: var(--background-light);
            color: var(--text-color-light);
        }
        [data-theme="dark"] .stApp {
            background: var(--background-dark);
            color: var(--text-color-dark);
        }

        /* Header Styling */
        .header-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            color: var(--text-color-light);
        }
        [data-theme="dark"] .header-title {
            color: var(--text-color-dark);
        }

        /* Input Box Styling */
        .stTextInput > div > div {
            background-color: var(--input-bg-light);
            color: var(--input-text-light);
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        [data-theme="dark"] .stTextInput > div > div {
            background-color: var(--input-bg-dark);
            color: var(--input-text-dark);
        }

        .stTextInput input::placeholder {
            color: var(--placeholder-light);
        }
        [data-theme="dark"] .stTextInput input::placeholder {
            color: var(--placeholder-dark);
        }

        /* Button Styling */
        .stButton button {
            background-color: #FF8C00;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-size: 1.1rem;
            border: none;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #FFA500;
        }

        /* Footer Styling */
        .footer {
            text-align: center;
            margin-top: 2rem;
            font-size: 0.9rem;
            color: var(--text-color-light);
        }
        [data-theme="dark"] .footer {
            color: var(--text-color-dark);
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
