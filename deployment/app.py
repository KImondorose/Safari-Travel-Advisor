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

# Updated function for dynamic styles
def add_dynamic_styles():
    st.markdown("""
        <style>
        :root {
            --bg-color: white;
            --text-color: black;
            --input-bg-light: white;
            --input-text-color-light: black;
            --input-placeholder-color-light: black;
            --input-bg-dark: white;
            --input-text-color-dark: white;
            --input-placeholder-color-dark: black;
            --card-bg-light: rgba(255, 255, 255, 0.85);
            --card-border-light: #e0e0e0;
            --card-bg-dark: rgba(50, 50, 50, 0.85);
            --card-border-dark: #555555;
            --button-bg: #FF8C00;
            --button-hover-bg: #FFA500;
            --button-text-light: white;
            --button-text-dark: black;
        }

        /* App background with image */
        .stApp {
            background: linear-gradient(to bottom, var(--bg-color), var(--bg-color)),
                        url("https://images.unsplash.com/photo-1501785888041-af3ef285b470?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") no-repeat center center fixed;
            background-size: cover;
        }

        /* Header styling */
        .header-title {
            color: var(--text-color);
            font-size: 3rem !important;
            font-weight: bold !important;
            text-align: center;
            margin: 1rem 0;
        }

        /* Input field styling for light and dark modes */
        [data-theme="light"] .stTextInput > div > div {
            background: var(--input-bg-light) !important;
            color: var(--input-text-color-light) !important;
            border: 2px solid var(--card-border-light);
            border-radius: 15px;
            transition: all 0.3s ease;
        }

        [data-theme="dark"] .stTextInput > div > div {
            background: var(--input-bg-dark) !important;
            color: var(--input-text-color-dark) !important;
            border: 2px solid var(--card-border-dark);
            border-radius: 15px;
            transition: all 0.3s ease;
        }

        /* Focus styling for input */
        .stTextInput > div > div:focus-within {
            border-color: var(--button-bg);
            box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.2);
        }

        /* Input text and placeholder styling */
        [data-theme="light"] .stTextInput input {
            background: var(--input-bg-light) !important;
            color: var(--input-text-color-light) !important;
        }

        [data-theme="dark"] .stTextInput input {
            background: var(--input-bg-dark) !important;
            color: var(--input-text-color-dark) !important;
        }

        [data-theme="light"] .stTextInput input::placeholder {
            color: var(--input-placeholder-color-light) !important;
        }

        [data-theme="dark"] .stTextInput input::placeholder {
            color: var(--input-placeholder-color-dark) !important;
        }

        /* Button styling */
        .stButton > button {
            background: var(--button-bg);
            color: var(--button-text-light);
            border: none;
            border-radius: 25px;
            font-size: 1.2rem;
            font-weight: bold;
            padding: 0.5rem 2rem;
            transition: background 0.3s ease, transform 0.2s ease;
        }

        .stButton > button:hover {
            background: var(--button-hover-bg);
            transform: translateY(-2px);
        }

        /* Card styling for light and dark modes */
        [data-theme="light"] .card {
            background: var(--card-bg-light);
            color: var(--text-color);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 1rem;
            border-left: 5px solid var(--button-bg);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        [data-theme="dark"] .card {
            background: var(--card-bg-dark);
            color: var(--text-color);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 1rem;
            border-left: 5px solid var(--button-bg);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .card h3 {
            color: var(--button-bg);
            margin-bottom: 10px;
        }

        .card p {
            color: var(--text-color);
        }

        /* Footer styling */
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            border-top: 1px solid var(--card-border-light);
        }

        .footer a {
            color: var(--button-bg);
            text-decoration: none;
            font-weight: bold;
        }

        .footer a:hover {
            color: var(--button-hover-bg);
        }
        </style>
    """, unsafe_allow_html=True)


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

# Function to preprocess text using NLTK lemmatizer
def preprocess_text(text):
    """
    Input raw text.
    Return preprocessed and lemmatized text using NLTK.
    """
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words containing numbers

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return [" ".join(lemmatized_tokens)]

final_model = joblib.load('/mount/src/travel-wordfinder/deployment/final_model.pkl')
vectorizer_final = joblib.load('/mount/src/travel-wordfinder/deployment/vectorizer_final.pkl')

add_dynamic_styles()
add_header()

st.markdown("""
    <div style='text-align: right; font-size: 1.2rem; font-weight: bold; margin-bottom: 2rem; color: var(--text-color);'>
        Discover your perfect destination! Share your dream activities, and we'll suggest the ideal places tailored just for you.
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_query = st.TextInput(
        "**What's your ideal travel experience?**",
        placeholder="E.g., 'Alpine meadows and glaciers, wildlife viewing...'"
    ).strip()

    predict_button = st.button("🔍 Find My Perfect Destination", use_container_width=True)

if predict_button:
    if user_query:
        try:
            processed_input = preprocess_text(user_query)
            predicted_country = final_model.predict(vectorizer_final.transform(processed_input))[0]
            
            st.markdown(f"""
                <div style='text-align: center; padding: 2rem;'>
                    <h2 style='color: var(--button-bg); margin-bottom: 1rem;'>Your Perfect Destination:</h2>
                    <h1 style='color: var(--button-bg); font-size: 3rem; margin-bottom: 2rem;'>{predicted_country} 🎯</h1>
                </div>
            """, unsafe_allow_html=True)

            filtered_data = preprocessed_df[preprocessed_df['Country'] == predicted_country]
            
            if not filtered_data.empty:
                vectorizer_attractions = TfidfVectorizer(stop_words='english')
                filtered_tfidf = vectorizer_attractions.fit_transform(filtered_data['Description'])
                query_tfidf = vectorizer_attractions.transform([user_query])
                
                similarity_scores = cosine_similarity(query_tfidf, filtered_tfidf).flatten()
                filtered_data['Similarity'] = similarity_scores
                
                top_attractions = filtered_data.sort_values(by='Similarity', ascending=False).head(5)
                
                st.markdown("<h2 style='color: var(--button-bg); text-align: center;'>Recommended Experiences</h2>", unsafe_allow_html=True)
                
                for idx, row in top_attractions.iterrows():
                    st.markdown(f"""
                        <div class="card">
                            <h3>✨ {row['Attraction']}</h3>
                            <p>{row['Description']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No attractions found for this destination yet.")
                
        except Exception as e:
            st.error("Oops! Something went wrong. Please try again.")
            print(f"Error: {str(e)}")
    else:
        st.warning("Please share your travel preferences first!")

st.markdown("""
    <div class="footer">
        <p>
            View code on 
            <a href="https://github.com/KImondorose/Travel-WordFinder" target="_blank">GitHub</a>
        </p>
    </div>
""", unsafe_allow_html=True)
