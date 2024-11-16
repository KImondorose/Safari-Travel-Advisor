import streamlit as st
import spacy
import re
import string
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add custom CSS for styling without background image
def add_styles():
    st.markdown(
        f"""
        <style>
        .header-title {{
            color: #4CAF50;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
        }}
        .header-icon {{
            font-size: 3em;
            color: #4CAF50;
        }}
        .card {{
            background-color: #F5F5F5;
            padding: 15px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin: 0;
            color: #4CAF50;
        }}
        .card p {{
            margin: 5px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add header with icon and title
def add_header():
    st.markdown("""
        <div>
            <span class="header-icon">🌍</span>
            <h1 class="header-title">Travel Destination Predictor</h1>
        </div>
    """, unsafe_allow_html=True)

# Load pre-trained model and vectorizer
final_model = joblib.load('final_model.pkl')
vectorizer_final = joblib.load('vectorizer_final.pkl')

# Load preprocessed dataset dynamically
@st.cache_data
def load_preprocessed_data():
    return pd.read_csv('preprocessed_df.csv')

preprocessed_df = load_preprocessed_data()

# Function to preprocess text
def preprocess_text(text):
    """
    Input raw text.
    Return preprocessed text.
    """
    nlp = spacy.load('en_core_web_sm')
    preprocessed = nlp(text)

    preprocessed = text.lower()
    preprocessed = re.sub('[%s]' % re.escape(string.punctuation), '', preprocessed)
    preprocessed = re.sub('\w*\d\w*', '', preprocessed)
    return [preprocessed]

# Add custom styles
add_styles()

# Add header
add_header()

# App layout
st.subheader("Input Preferences")
user_query = st.text_input("Enter your travel preferences (e.g.,'Alpine meadows and glaciers') and we'll predict the best country and attraction for you!:").strip()

if st.button("Predict"):
    if user_query:
        # Step 1: Preprocess the input query
        processed_input = preprocess_text(user_query)
        
        # Step 2: Predict the country
        try:
            predicted_country = final_model.predict(vectorizer_final.transform(processed_input))[0]
            st.write(f"Predicted Country: {predicted_country}")
            
            # Step 3: Filter attractions by the predicted country
            filtered_data = preprocessed_df[preprocessed_df['Country'] == predicted_country]

            if filtered_data.empty:
                st.warning("No attractions found for the predicted country.")
            else:
                # Step 4: Rank Attractions by Similarity
                vectorizer_attractions = TfidfVectorizer(stop_words='english')
                filtered_tfidf = vectorizer_attractions.fit_transform(filtered_data['Description'])
                query_tfidf = vectorizer_attractions.transform([user_query])

                # Calculate similarity
                similarity_scores = cosine_similarity(query_tfidf, filtered_tfidf).flatten()
                filtered_data['Similarity'] = similarity_scores

                # Step 5: Output Top Attractions
                top_attractions = filtered_data.sort_values(by='Similarity', ascending=False).head(5)

                st.subheader("Recommended Attractions:")
                for idx, row in top_attractions.iterrows():
                    st.markdown(f"""
                        <div class="card">
                            <h3>{row['Attraction']}</h3>
                            <p>{row['Description']}</p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error predicting country: {e}")
    else:
        st.warning("Please enter a query before pressing Predict.")

# Add footer
st.markdown("""
    <hr>
    <div style="text-align: center; font-size: 0.9em; color: #888;">
        Created with ❤️ by <a href="https://github.com/KImondorose/Travel-WordFinder" target="_blank" style="color: #4CAF50; text-decoration: none;">Group 10</a>
    </div>
""", unsafe_allow_html=True)
