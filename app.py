import logging
from collections import OrderedDict
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import pandas as pd
from flask_caching import Cache
from datetime import datetime
import matplotlib.pyplot as plt
import io
import os
import torch
import pickle
from model import CollaborativeFilteringModel
from data_loader import get_data_loader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
import random
import markdown
import requests

app = Flask(__name__)

# Configure cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'}) # Can be 'simple', 'redis', 'memcached', etc.

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO) # Change to DEBUG for more verbosity

# Global variable to store the product names
product_names = {} 


@app.before_request
def start_timer():
    request.start_time = datetime.now() # Start the timer

# Log details of each request
@app.after_request
def log_request(response):
    if not hasattr(request, 'start_time'):
        return response
    duration = datetime.now() - request.start_time
    log_details = {
        'method': request.method,
        'path': request.path,
        'status': response.status_code,
        'duration': duration.total_seconds(),
        'timestamp': datetime.now().isoformat()
    }
    app.logger.info(log_details)
    return response

# Configuration
USERS_PER_PAGE = 10

# Load the dataset for user listing
data_path = 'dataset/Reviews.csv' # Reviews dataset
tfidf_matrix_path = 'dataset/tfidf_matrix.pkl' # TF-IDF matrix
svd_matrix_path = 'dataset/svd_matrix.pkl' # SVD matrix
faiss_index_path = 'dataset/faiss.index' # Faiss index

# Path for storing new reviews
new_reviews_file = 'dataset/new_reviews.csv' # New reviews file

def download_files(urls, destinations):
    """
    Downloads files from the given URLs and saves them to the specified destinations.

    :param urls: List of URLs of the files to download.
    :param destinations: List of local paths where the files will be saved.
    """
    if len(urls) != len(destinations):
        print("The number of URLs and destinations must be the same.")
        return

    for url, destination in zip(urls, destinations):
        # Check if the file already exists
        if os.path.exists(destination):
            print(f"File already exists: {destination}. Skipping download.")
            continue

        try:
            # Send a GET request to the URL
            response = requests.get(url, stream=True)
            # Check if the request was successful
            if response.status_code == 200:
                # Create the necessary directories if they don't exist
                os.makedirs(os.path.dirname(destination), exist_ok=True)

                # Open the local file in write-binary mode
                with open(destination, 'wb') as file:
                    # Write the content of the response to the local file in chunks
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"File successfully downloaded and saved to {destination}")
            else:
                print(f"Failed to download file from {url}. HTTP Status Code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while downloading {url}: {e}")


# Example usage:
urls = [
    'https://aimodelaws.s3.amazonaws.com/svd_matrix.pkl',
    'https://aimodelaws.s3.amazonaws.com/tfidf_matrix.pkl',
    'https://aimodelaws.s3.amazonaws.com/reco_model.pth',
    'https://aimodelaws.s3.amazonaws.com/Reviews.csv'
]

destinations = [
    'dataset/svd_matrix.pkl',
    'dataset/tfidf_matrix.pkl',
    'models/reco_model.pth',
    'dataset/Reviews.csv'
]

# Function to initialize the data and model
def initialize():
    global df, model, faiss_index, reduced_tfidf_matrix, num_users, num_items, user_to_idx, idx_to_user, item_to_idx, idx_to_item, device, popular_items, product_names

    download_files(urls, destinations) # Download the required files

    # Load data
    try:
        df = pd.read_csv(data_path) # Load the dataset
    except FileNotFoundError:
        logging.error("Dataset not found at %s", data_path) 
        df = None
        return

    # Load product names if available
    product_names_path = 'dataset/ProductNames.csv' # Product names file
    
    if os.path.exists(product_names_path):
        product_names_df = pd.read_csv(product_names_path)
        product_names = pd.Series(product_names_df.ProductName.values, index=product_names_df.ProductId).to_dict()
        logging.info("Loaded product names from %s", product_names_path)
    else:
        logging.warning("Product names file not found at %s", product_names_path)

    # Log the columns and the first few rows for debugging
    logging.info("Columns in the DataFrame: %s", df.columns)
    logging.info("First few rows of the DataFrame: %s", df.head())

    # Handle missing values using forward fill
    df.ffill(inplace=True)

    # Normalize/clean data by removing duplicates
    df.drop_duplicates(inplace=True)

    # Check if columns exist before downcasting
    columns_to_downcast = ['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time']
    for col in columns_to_downcast:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Add this block to create the 'combined_text' column
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')
    df['combined_text'] = df['Summary'] + ' ' + df['Text']

    # Load the trained PyTorch model
    model_path = 'models/reco_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the checkpoint to get the original num_users and num_items
    checkpoint = torch.load(model_path, map_location=device)
    num_users = checkpoint['model_state_dict']['user_embedding.weight'].size(0)
    num_items = checkpoint['model_state_dict']['item_embedding.weight'].size(0)

    # DataLoader
    batch_size = 64  # Keep it consistent with training
    data_loader, _, _, user_to_idx, idx_to_user, item_to_idx, idx_to_item = get_data_loader(df, batch_size)

    # Initialize the model with the original dimensions
    model = CollaborativeFilteringModel(num_users, num_items, embedding_dim=32).to(device)

    # Load the checkpoint and extract the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Check and load or compute TF-IDF, SVD, and Faiss index
    if os.path.exists(tfidf_matrix_path) and os.path.exists(svd_matrix_path) and os.path.exists(faiss_index_path):
        logging.info("Loading precomputed TF-IDF, SVD, and Faiss index...")
        with open(tfidf_matrix_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(svd_matrix_path, 'rb') as f:
            reduced_tfidf_matrix = pickle.load(f)
        faiss_index = faiss.read_index(faiss_index_path)
    else:
        # Compute TF-IDF, SVD, and Faiss index, and save them for future use  
        logging.info("Computing TF-IDF, SVD, and Faiss index...")
        # TF-IDF for content-based filtering with Faiss
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])

        # Perform dimensionality reduction using TruncatedSVD
        svd = TruncatedSVD(n_components=100, random_state=42)
        reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

        # Convert to float32
        reduced_tfidf_matrix = reduced_tfidf_matrix.astype('float32')

        # Use Faiss to create an index for approximate nearest neighbors
        faiss.normalize_L2(reduced_tfidf_matrix)
        faiss_index = faiss.IndexFlatL2(reduced_tfidf_matrix.shape[1])
        faiss_index.add(reduced_tfidf_matrix)

        # Save the precomputed data
        with open(tfidf_matrix_path, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(svd_matrix_path, 'wb') as f:
            pickle.dump(reduced_tfidf_matrix, f)
        faiss.write_index(faiss_index, faiss_index_path)

    logging.info("TF-IDF, SVD, and Faiss index are ready.")

    # Popular items based on average scores
    popular_items = df.groupby('ProductId')['Score'].mean().sort_values(ascending=False).index.tolist()
    logging.info("Computed popular items")

# Initialize the data and model outside of the main block
initialize()

# Function to generate top N recommendations for a user, picking a random subset for display
def get_recommendations(user_id, top_n=20, display_n=5):
    if user_id not in user_to_idx:
        logging.warning("User ID %s not found in user_to_idx mapping.", user_id)
        return []

    user_id_idx = user_to_idx[user_id]
    if user_id_idx >= num_users:
        logging.error("User index %s is out of range for num_users %s.", user_id_idx, num_users)
        return []

    item_ids = torch.arange(num_items).to(device)
    user_ids = torch.tensor([user_id_idx] * num_items).to(device)

    with torch.no_grad():
        scores = model(user_ids, item_ids).squeeze()

    top_scores, top_items = torch.topk(scores, top_n)
    recommendations = []
    for item_id, score in zip(top_items, top_scores):
        item_idx = item_id.item()
        if item_idx in idx_to_item:
            product_id = idx_to_item[item_idx]
            summary = df[df['ProductId'] == product_id]['Summary'].values[0]
            product_name = product_names.get(product_id, summary)  # Use product name if available, otherwise fallback to summary
            recommendations.append((product_id, score.item(), product_name))
        else:
            logging.warning("Item ID %s not found in idx_to_item mapping.", item_idx)

    # Randomly select display_n products from top_n recommendations
    if len(recommendations) > display_n:
        recommendations = random.sample(recommendations, display_n)

    logging.info("Recommendations for user %s: %s", user_id, recommendations)
    return recommendations



# Function to get popular items
def get_popular_items(top_n=20):
    popular_items = df.groupby('ProductId')['Score'].mean().sort_values(ascending=False).index[:top_n]
    popular_recs = []
    for pid in popular_items:
        avg_rating = df[df['ProductId'] == pid]['Score'].mean()
        summary = df[df['ProductId'] == pid]['Summary'].values[0] if not df[df['ProductId'] == pid].empty else "No summary available"
        product_name = product_names.get(pid, summary)
        popular_recs.append((pid, avg_rating, product_name))
    return popular_recs

# Function for content-based recommendations
def content_based_recommendations(product_id, n=10):
    if product_id not in item_to_idx:
        logging.warning("Product ID %s not found in item_to_idx mapping.", product_id)
        return []

    idx = item_to_idx[product_id]
    distances, indices = faiss_index.search(reduced_tfidf_matrix[idx].reshape(1, -1), n + 1)  # +1 because the first result is the query item itself
    return df['ProductId'].iloc[indices.flatten()[1:]].tolist()

# Function for hybrid recommendations, combining collaborative and content-based filtering
def hybrid_recommendations(user_id, top_n=20, display_n=5):
    if user_id not in user_to_idx:
        logging.warning("User %s not found, returning popular items.", user_id)
        return get_popular_items(display_n)

    collaborative_recs = get_recommendations(user_id, top_n, display_n)
    if not collaborative_recs:
        return get_popular_items(display_n)

    collaborative_product_ids = {pid: (pid, score, summary) for pid, score, summary in collaborative_recs}

    hybrid_recs = OrderedDict()

    # Add collaborative recommendations
    for pid, (product_id, score, summary) in collaborative_product_ids.items():
        if len(hybrid_recs) < display_n:
            hybrid_recs[product_id] = (product_id, score, product_names.get(product_id, summary))

    # Add content-based recommendations ensuring no duplicates
    for product_id in collaborative_product_ids.keys():
        content_recs = content_based_recommendations(product_id, top_n // 2)
        for content_pid in content_recs:
            if content_pid not in hybrid_recs and len(hybrid_recs) < display_n:
                content_summary = df[df['ProductId'] == content_pid]['Summary'].values[0] if not df[df['ProductId'] == content_pid].empty else "No summary available"
                hybrid_recs[content_pid] = (content_pid, collaborative_product_ids[product_id][1], product_names.get(content_pid, content_summary))  # Use collaborative score

    # Supplement with popular items to meet the quota of display_n
    if len(hybrid_recs) < display_n:
        popular_recs = get_popular_items(display_n - len(hybrid_recs))
        for popular_pid, avg_rating, popular_summary in popular_recs:
            if popular_pid not in hybrid_recs and len(hybrid_recs) < display_n:
                hybrid_recs[popular_pid] = (popular_pid, avg_rating, popular_summary)

    resolved_recommendations = list(hybrid_recs.values())

    # Randomly select display_n items from resolved recommendations
    if len(resolved_recommendations) > display_n:
        final_recommendations = random.sample(resolved_recommendations, display_n)
    else:
        final_recommendations = resolved_recommendations

    logging.info("Hybrid recommendations for user %s: %s", user_id, final_recommendations)
    return final_recommendations

@app.context_processor
def utility_processor():
    def range_to(start, end):  
        return range(start, end)

    def min_value(a, b):
        return min(a, b)

    def max_value(a, b):
        return max(a, b)

    return dict(range_to=range_to, min_value=min_value, max_value=max_value)

# The main route, which lists users, and allows for pagination
@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):

    if df is None:
        return "The application requires dataset/Reviews.csv to function. Please upload the file."

    unique_users = df[['UserId', 'ProfileName']].drop_duplicates()
    num_users = len(unique_users)
    total_pages = (num_users // USERS_PER_PAGE) + (1 if num_users % USERS_PER_PAGE != 0 else 0)

    users = unique_users.iloc[(page-1)*USERS_PER_PAGE:page*USERS_PER_PAGE]

    return render_template('index.html', users=users, page=page, total_pages=total_pages)

# Route to display the recommendations for a user
@app.route('/recommendations/<user_id>')
@cache.cached(timeout=600, key_prefix='recommendations_%s')
def recommendations(user_id):
    top_recommendations = hybrid_recommendations(user_id, top_n=20, display_n=5)
    return render_template('recommendations.html', user_id=user_id, recommendations=top_recommendations)

# Route to collect feedback from users
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    feedback = data.get('feedback')

    # Convert feedback to score
    if feedback == 'like':
        score = 5
    elif feedback == 'dislike':
        score = 1
    else:
        return jsonify({"status": "error", "message": "Invalid feedback type"}), 400

    logging.info("Received feedback for user %s on product %s: %s", user_id, product_id, feedback)

    # Prepare feedback data in the same structure as Reviews.csv
    feedback_data = pd.DataFrame({
        'Id': [None],  # Assigning None for this transaction
        'ProductId': [product_id],
        'UserId': [user_id],
        'ProfileName': [None],  # Assuming ProfileName is not available
        'HelpfulnessNumerator': [None],
        'HelpfulnessDenominator': [None],
        'Score': [score],
        'Time': [datetime.now().timestamp()],
        'Summary': [''],  # Assuming no summary for feedback
        'Text': [feedback]  # Storing original feedback text
    })

    if os.path.exists(new_reviews_file):
        feedback_data.to_csv(new_reviews_file, mode='a', header=False, index=False)
    else:
        feedback_data.to_csv(new_reviews_file, index=False)

    return jsonify({"status": "success"}), 200

@app.route('/readme')
def readme():
    with open('README.MD', 'r') as file:
        content = file.read()
    return render_template('readme.html', content=content)


# Route that displays the deep learning training metrics
@app.route('/training_metrics')
def training_metrics():
    metrics_file_path = 'models/training_metrics.csv'
    if not os.path.exists(metrics_file_path):
        return "Training metrics not found", 404

    metrics_df = pd.read_csv(metrics_file_path)

    plt.figure(figsize=(18, 12))

    # Plot training loss
    plt.subplot(2, 3, 1)
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot training accuracy
    plt.subplot(2, 3, 2)
    plt.plot(metrics_df['epoch'], metrics_df['train_accuracy'], marker='o')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Plot test loss
    plt.subplot(2, 3, 3)
    plt.plot(metrics_df['epoch'], metrics_df['test_loss'], marker='o')
    plt.title('Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot test accuracy
    plt.subplot(2, 3, 4)
    plt.plot(metrics_df['epoch'], metrics_df['test_accuracy'], marker='o')
    plt.title('Test Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Plot precision
    plt.subplot(2, 3, 5)
    plt.plot(metrics_df['epoch'], metrics_df['precision'], marker='o')
    plt.title('Precision Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True)

    # Plot recall
    plt.subplot(2, 3, 6)
    plt.plot(metrics_df['epoch'], metrics_df['recall'], marker='o')
    plt.title('Recall Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)