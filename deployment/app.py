
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
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

def add_dynamic_styles():
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Main background with parallax effect */
        .stApp {
        background: url("https://images.unsplash.com/photo-1731048935114-4b84ba084619?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")no-repeat center center fixed;
        background-size: cover; /* Ensures the image covers the entire screen */
        background-attachment: fixed; /* Parallax effect */
        margin: 0;
        padding: 0;
        }
        
        /* Header styling */
        .header-title {
            background: linear-gradient(135deg, #FF8C00, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            text-align: center;
            padding: 2rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Card styling */
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 25px 0;
            transition: all 0.3s ease;
            border-left: 5px solid #FF8C00;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .card h3 {
            color: #FF8C00;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #FF8C00, #FFA500);
            color: white;
            border: none;
            padding: 1rem 2.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            width: 100%;
            max-width: 300px;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255,140,0,0.3);
        }
        
        /* Input field styling */
        .stTextInput > div > div {
            border-radius: 15px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
            background: white;
        }
        
        .stTextInput > div > div:focus-within {
            border-color: #FF8C00;
            box-shadow: 0 0 0 3px rgba(255,140,0,0.2);
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            border-top: 1px solid rgba(0,0,0,0.1);
        }
        
        .footer a {
            color: #FF8C00;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        
        .footer a:hover {
            color: #FFA500;
        }
        
        .stTextInput > div > div {
            border-radius: 15px;
            border: 2px solid var(--card-border);
            transition: all 0.3s ease;
            background: var(--card-bg) !important;
            color: var(--text-color) !important;
        }

        .stTextInput input {
            color: black !important;
                
        }

        /* For text area if you're using it */
            .stTextArea textarea {
            color: var(--text-color) !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .header-title {
                font-size: 2.5rem !important;
            }
            .card {
                padding: 15px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def add_header():
    st.markdown("""
        <div>
            <h1 class="header-title">🌍 Safari Travel Advisor</h1>
        </div>
    """, unsafe_allow_html=True)

# [Rest of your existing code remains the same until the app layout section]

# App layout
add_dynamic_styles()
add_header()

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

st.markdown("""
    <div style='text-align: right; color: black; font-size: 1.2rem; font-weight: bold; margin-bottom: 2rem;'>
        Discover your perfect destination! Share your dream activities, and we'll suggest the ideal places tailored just for you. 
            Try mentioning activities like "hiking mountain trails", "exploring ancient ruins", or "relaxing on beaches."
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    user_query = st.text_input(
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
                    <h2 style='color: #FF8C00; margin-bottom: 1rem;'>Your Perfect Safari Destination:</h2>
                    <h1 style='color: #FF8C00; font-size: 3rem; margin-bottom: 2rem;'>{predicted_country} 🎯</h1>
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
                
                st.markdown("<h2 style='color: #FF8C00; text-align: center;'>Recommended Experiences</h2>", unsafe_allow_html=True)
                
                for idx, row in top_attractions.iterrows():
                    st.markdown(f"""
                        <div class="card">
                            <h3>✨ {row['Attraction']}</h3>
                            <p style='color: #444; line-height: 1.6;'>{row['Description']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No attractions found for this destination yet.")
                
        except Exception as e:
            st.error("Oops! Something went wrong. Please try again.")
            print(f"Error: {str(e)}")  # For debugging
    else:
        st.warning("Please share your travel preferences first!")

st.markdown("""
    <div class="footer">
        <p style="color: red; font-weight: bold;">
            View code on 
            <a href="https://github.com/KImondorose/Travel-WordFinder" target="_blank" style="color: red; font-weight: bold; text-decoration: underline;">
                <u>GitHub</u>
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)
