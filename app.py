from flask import Flask, render_template, request
import pandas as pd
import joblib
from svd_tqdm import SVDTqdm

app = Flask(__name__)

# Configuration
USERS_PER_PAGE = 10

# Load the dataset for user listing
data_path = 'dataset/Reviews.csv'
df = pd.read_csv(data_path)

# Handle missing values using forward fill
df.ffill(inplace=True)

# Normalize/clean data by removing duplicates
df.drop_duplicates(inplace=True)

# Downcast numeric types to save memory
df['HelpfulnessNumerator'] = pd.to_numeric(df['HelpfulnessNumerator'], downcast='integer')
df['HelpfulnessDenominator'] = pd.to_numeric(df['HelpfulnessDenominator'], downcast='integer')
df['Score'] = pd.to_numeric(df['Score'], downcast='integer')
df['Time'] = pd.to_numeric(df['Time'], downcast='integer')

# Load the trained SVD model and product info
model = joblib.load('models/svd_model.pkl')
product_info = joblib.load('models/product_info.pkl')

# Function to generate top N recommendations for a user
def get_top_n_recommendations(user_id, model, df, product_info, n=5):
    # Get the list of all product IDs
    all_product_ids = df['ProductId'].unique()

    # Get the IDs of products the user has already rated
    rated_products = df[df['UserId'] == user_id]['ProductId'].tolist()

    # Predict ratings for all products not yet rated by the user
    unrated_products = [product for product in all_product_ids if product not in rated_products]
    predictions = [model.predict(user_id, product_id) for product_id in unrated_products]

    # Sort predictions by estimated rating in descending order
    top_n_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    # Return the top N product recommendations with summaries
    return [(pred.iid, pred.est, product_info.get(pred.iid, "No summary available")) for pred in top_n_predictions]

@app.context_processor
def utility_processor():
    def range_to(start, end):
        return range(start, end)

    def min_value(a, b):
        return min(a, b)

    def max_value(a, b):
        return max(a, b)

    return dict(range_to=range_to, min_value=min_value, max_value=max_value)

@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
    unique_users = df[['UserId', 'ProfileName']].drop_duplicates()
    num_users = len(unique_users)
    total_pages = (num_users // USERS_PER_PAGE) + (1 if num_users % USERS_PER_PAGE != 0 else 0)

    users = unique_users.iloc[(page-1)*USERS_PER_PAGE:page*USERS_PER_PAGE]

    return render_template('index.html', users=users, page=page, total_pages=total_pages)

@app.route('/recommendations/<user_id>')
def recommendations(user_id):
    top_recommendations = get_top_n_recommendations(user_id, model, df, product_info, n=5)
    return render_template('recommendations.html', user_id=user_id, recommendations=top_recommendations)

if __name__ == '__main__':
    app.run(debug=True)