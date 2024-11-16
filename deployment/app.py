from text_preprocessor import PreprocessText
from flask import Flask, request, jsonify
import pickle

# Load the pipeline
with open('destination_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get('text')
    if not user_input:
        return jsonify({'error': 'No input text provided'}), 400

    # Make prediction using the pipeline
    predicted_country = pipeline.predict([user_input])
    return jsonify({'predicted_country': predicted_country[0]})

if __name__ == '__main__':
    app.run(debug=True)