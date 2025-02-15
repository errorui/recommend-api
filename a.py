import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
rec = pd.read_csv('dataset.csv')  # Ensure dataset.csv is in the same folder

# Convert text data to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(rec['Sector']).toarray()
similarity = cosine_similarity(vectors)

# Create Flask app
app = Flask(__name__)

# Recommendation function
def recommend(stock):
    try:
        stock_index = rec[rec['Company Name'] == stock].index[0]
        distances = similarity[stock_index]
        stock_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
        
        recommendations = [(rec.iloc[i[0]]['Company Name'], rec.iloc[i[0]]['Ticker']) for i in stock_list]
        return recommendations
    except IndexError:
        return ["Stock not found"]

# Define API route
@app.route('/recommend', methods=['GET'])
def recommend_stock():
    stock = request.args.get('stock')
    if not stock:
        return jsonify({"error": "No stock name provided"}), 400

    recommendations = recommend(stock)
    return jsonify({"recommended_stocks": recommendations})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)