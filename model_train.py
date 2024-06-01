import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from model import CollaborativeFilteringModel
from data_loader import get_data_loader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
from sklearn.metrics import precision_score, recall_score

# Verify CUDA
try:
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
except Exception as e:
    print(f"Error checking CUDA: {e}")

# Hyperparameters
embedding_dim = 32
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-5  # L2 regularization term
step_size = 7  # Reduce learning rate every 5 epochs
gamma = 0.1    # Reduce the learning rate by a factor of 0.1
num_epochs = 20
dropout_rate = 0.2
csv_file = 'dataset/Reviews.csv'
new_reviews_file = 'dataset/new_reviews.csv'
model_dir = 'models'
model_path = os.path.join(model_dir, 'reco_model.pth')
metrics_path = os.path.join(model_dir, 'training_metrics.csv')

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Check for GPU
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
except Exception as e:
    print(f"Error setting device: {e}")
    device = torch.device('cpu')

# Load data and integrate new reviews
try:
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    if os.path.exists(new_reviews_file):
        new_df = pd.read_csv(new_reviews_file)
        print(f"Loaded {len(new_df)} new reviews from {new_reviews_file}")
        df = pd.concat([df, new_df], ignore_index=True)
except Exception as e:
    print(f"Error loading data: {e}")
    raise e

# Preprocessing
try:
    df['combined_text'] = df['Summary'].fillna('') + ' ' + df['Text'].fillna('')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Split data into {len(train_df)} training rows and {len(test_df)} test rows")
except Exception as e:
    print(f"Error during preprocessing and splitting: {e}")
    raise e

# DataLoader for training set
try:
    train_loader, num_users, num_items, user_to_idx, idx_to_user, item_to_idx, idx_to_item = get_data_loader(train_df, batch_size)
    print(f"Created DataLoader for training set with {num_users} users and {num_items} items")
except Exception as e:
    print(f"Error creating DataLoader for training set: {e}")
    raise e

# DataLoader for test set
try:
    test_loader, _, _, _, _, _, _ = get_data_loader(test_df, batch_size)
    print(f"Created DataLoader for test set")
except Exception as e:
    print(f"Error creating DataLoader for test set: {e}")
    raise e

# TF-IDF for content-based filtering with Faiss
try:
    print("Initializing TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words='english')
    print("Fitting TF-IDF Vectorizer...")

    # Fit and transform the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    print(f"TF-IDF matrix shape before reduction: {tfidf_matrix.shape}")

    # Perform dimensionality reduction using TruncatedSVD
    print("Performing dimensionality reduction with TruncatedSVD...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)
    print(f"TF-IDF matrix shape after reduction: {reduced_tfidf_matrix.shape}")

    # Convert to float32
    reduced_tfidf_matrix = reduced_tfidf_matrix.astype('float32')

    # Use Faiss to create an index for approximate nearest neighbors
    print("Building Faiss index...")
    faiss.normalize_L2(reduced_tfidf_matrix)
    index = faiss.IndexFlatL2(reduced_tfidf_matrix.shape[1])
    index.add(reduced_tfidf_matrix)

    print("Faiss index built successfully.")

except Exception as e:
    print(f"Error during TF-IDF computation: {e}")
    raise e

# Popular items based on average scores
popular_items = df.groupby('ProductId')['Score'].mean().sort_values(ascending=False).index.tolist()
print(f"Computed popular items")

# Function to save the model
def save_model(model, optimizer, epoch, loss, accuracy, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', float('inf'))
        accuracy = checkpoint.get('accuracy', 0.0)  # Use .get() to provide a default value if 'accuracy' key is missing
        model.to(device)
        print(f"Model loaded from {path}, epoch {epoch}, loss {loss}, accuracy {accuracy}")
        return model, optimizer, epoch, loss, accuracy
    else:
        print(f"No checkpoint found at {path}, starting from scratch.")
        return model, optimizer, 0, float('inf'), 0.0

# Initialize metrics logging
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    start_epoch = metrics_df['epoch'].max() + 1
    print(f"Metrics loaded, starting from epoch {start_epoch}")
else:
    metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'precision', 'recall'])
    start_epoch = 0
    print("Initialized new metrics DataFrame")

# Model, loss function, optimizer
try:
    model = CollaborativeFilteringModel(num_users, num_items, embedding_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("Model and optimizer initialized")
except Exception as e:
    print(f"Error initializing model and optimizer: {e}")
    raise e

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Load model if a checkpoint exists
try:
    model, optimizer, start_epoch_checkpoint, _, _ = load_model(model_path, model, optimizer)
    start_epoch = max(start_epoch, start_epoch_checkpoint)  # Ensure we pick up at the correct epoch
    print(f"Model loaded, starting from epoch {start_epoch}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

criterion = nn.MSELoss()

print("Starting training loop...")

# Training loop
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} - Training...")
    total_train_loss = 0
    correct_train_predictions = 0
    total_train_predictions = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
    for user_id, item_id, score in progress_bar:
        user_id = user_id.long().to(device)
        item_id = item_id.long().to(device)
        score = score.float().to(device)

        optimizer.zero_grad()
        output = model(user_id, item_id).squeeze()
        loss = criterion(output, score)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Calculate accuracy (rounding predictions to nearest integer)
        correct_train_predictions += torch.sum(torch.round(output) == score).item()
        total_train_predictions += score.size(0)

        avg_train_loss = total_train_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=avg_train_loss)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train_predictions / total_train_predictions
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy}')

    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Evaluating...")

    # Evaluate the model on the test set
    model.eval()
    total_test_loss = 0
    correct_test_predictions = 0
    total_test_predictions = 0

    with torch.no_grad():
        for user_id, item_id, score in test_loader:
            user_id = user_id.long().to(device)
            item_id = item_id.long().to(device)
            score = score.float().to(device)

            output = model(user_id, item_id).squeeze()
            loss = criterion(output, score)
            total_test_loss += loss.item()

            # Calculate accuracy (rounding predictions to nearest integer)
            correct_test_predictions += torch.sum(torch.round(output) == score).item()
            total_test_predictions += score.size(0)

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = correct_test_predictions / total_test_predictions
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}')

    # Calculate Precision and Recall with a threshold
    # Assuming a threshold of 3 for a positive prediction
    y_true = (score.cpu().numpy() >= 3).astype(int)
    y_pred = (output.cpu().numpy() >= 3).astype(int)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')

    print(f'Epoch {epoch+1}/{num_epochs}, Precision: {precision}, Recall: {recall}')

    # Save the model after each epoch
    save_model(model, optimizer, epoch+1, avg_train_loss, train_accuracy, model_path)

    # Create a new DataFrame for the current epoch's metrics
    epoch_metrics_df = pd.DataFrame({
        'epoch': [epoch+1], 
        'train_loss': [avg_train_loss], 
        'train_accuracy': [train_accuracy],
        'test_loss': [avg_test_loss],
        'test_accuracy': [test_accuracy],
        'precision': [precision],
        'recall': [recall]
    })

    # Concatenate the new metrics DataFrame with the existing metrics DataFrame
    metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)

    # Save metrics to CSV
    metrics_df.to_csv(metrics_path, index=False)

# Plotting the training and test loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(metrics_df['epoch'], metrics_df['train_loss'], marker='o', label='Train Loss')
plt.plot(metrics_df['epoch'], metrics_df['test_loss'], marker='o', label='Test Loss', linestyle='--')
plt.title('Training and Test Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(metrics_df['epoch'], metrics_df['train_accuracy'], marker='o', label='Train Accuracy')
plt.plot(metrics_df['epoch'], metrics_df['test_accuracy'], marker='o', label='Test Accuracy', linestyle='--')
plt.plot(metrics_df['epoch'], metrics_df['precision'], marker='o', label='Precision', linestyle='-.')
plt.plot(metrics_df['epoch'], metrics_df['recall'], marker='o', label='Recall', linestyle=':')
plt.title('Training and Test Metrics over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the final model
save_model(model, optimizer, num_epochs, avg_train_loss, train_accuracy, model_path)

# Popular items function
def get_popular_items(n=10):
    return df.groupby('ProductId')['Score'].mean().sort_values(ascending=False).index[:n].tolist()

# Content-based recommendations function
def content_based_recommendations(product_id, n=10):
    idx = df[df['ProductId'] == product_id].index[0]
    distances, indices = index.search(reduced_tfidf_matrix[idx].reshape(1, -1), n + 1)  # +1 because the first result is the query item itself
    return df['ProductId'].iloc[indices.flatten()[1:]].tolist()

# Hybrid recommendations function
def hybrid_recommendations(user_id, n=10):
    if user_id not in user_to_idx:
        return get_popular_items(n)

    collaborative_recs = get_recommendations(user_id, n)
    hybrid_recs = []
    for product_id in collaborative_recs:
        content_recs = content_based_recommendations(product_id, n//2)
        hybrid_recs.extend(content_recs)

    hybrid_recs = list(set(hybrid_recs))
    if len(hybrid_recs) < n:
        hybrid_recs.extend(get_popular_items(n - len(hybrid_recs)))

    return hybrid_recs[:n]

# Function to get recommendations
def get_recommendations(user_id, n=10):
    model.eval()
    user_id = user_to_idx[user_id]
    item_ids = torch.arange(num_items).to(device)
    user_ids = torch.tensor([user_id] * num_items).to(device)

    with torch.no_grad():
        scores = model(user_ids, item_ids).squeeze()

    top_scores, top_items = torch.topk(scores, n)
    top_item_names = [idx_to_item[item_id.item()] for item_id in top_items]
    return top_item_names

# Example usage:
user_id = 'XYZ123'  # New user
recommendations = hybrid_recommendations(user_id, n=10)
print(f'Recommendations for user {user_id}: {recommendations}')

user_id = 'A1D87F6ZCVE5NK'  # Existing user
recommendations = hybrid_recommendations(user_id, n=10)
print(f'Recommendations for user {user_id}: {recommendations}')
