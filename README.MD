# AI-Powered Recommendation Engine

## Overview
This project implements an AI-powered recommendation engine for an e-commerce platform using a hybrid recommendation system that combines collaborative filtering and content-based filtering. The engine is built using PyTorch for the model, Faiss for efficient similarity search, and Flask to create a web application for providing personalized product recommendations.

## Features
- **Data Preprocessing**: Handles missing values and dataset cleaning.
- **Hybrid Recommendation System**: Combines collaborative filtering and content-based filtering.
- **Caching**: Uses Flask-Caching for low-latency responses.
- **Logging**: Logs request handling and performance metrics.
- **Feedback Collection**: Collects user feedback for continuous model evaluation and improvement.

## Setup
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/glenwrhodes/Crossover_AIModel.git
    cd Crossover_AIModel
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the Model**:
    ```bash
    python model_train.py
    ```

4. **Run the Flask Application**:
    ```bash
    python app.py
    ```

## Usage
- Access the application at `http://127.0.0.1:5000/`.
- Select a user to view personalized recommendations.
- Provide feedback on recommendations.

## Model Structure

### 1. Data Collection and Preprocessing

**Data Source**:
- The dataset used is Amazon review data. The relevant columns include:
  - `ProductId`: Identifies the product.
  - `UserId`: Identifies the user.
  - `Score`: The rating given by the user, which serves as the target variable.
  - `Summary` and `Text`: The review content utilized in content-based filtering.

**Data Integration**:
- Primary dataset is read from a CSV file (`Reviews.csv`). If new reviews are available, they are integrated from `new_reviews.csv` to continuously update and improve the model.

**Preprocessing Steps**:
- Combine `Summary` and `Text` into a single `combined_text` column to use in the content-based filtering step.

```python
# Example: Combining text columns
df['combined_text'] = df['Summary'].fillna('') + ' ' + df['Text'].fillna('')
```

### 2. Model Development

**Collaborative Filtering**:
- Utilizes a neural network model (CollaborativeFilteringModel) with embedding layers for users and items. This technique helps in capturing latent features from user-item interactions.

```python
class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout_rate=0.2):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, user_id, item_id):
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        user_bias = self.user_bias(user_id).squeeze()
        item_bias = self.item_bias(item_id).squeeze()

        # Apply dropout to embeddings
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)

        dot_product = (user_emb * item_emb).sum(1)
        return dot_product + user_bias + item_bias + self.global_bias
```

**Forward Method**:
- Calculates the dot product between user and item embeddings, adds bias terms, and applies dropout.

### 3. Data Loader

**Custom Dataset**:
- The `AmazonReviewDataset` class maps user and item IDs to numerical indices. This mapping facilitates the embedding process and allows efficient data handling during training.

**DataLoader**:
- The `get_data_loader` function creates data loaders for the training and testing datasets, enabling batched processing to speed up training and evaluation.

```python
# Example: Creating DataLoader
train_loader, num_users, num_items, user_to_idx, idx_to_user, item_to_idx, idx_to_item = get_data_loader(train_df, batch_size)
```

### 4. Training Process

**Hyperparameters**:
- **Embedding Dimension**: 32
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Weight Decay**: 1e-5 (L2 regularization term)
- **Dropout Rate**: 0.2
- **Number of Epochs**: 20

**Optimization**:
- **Adam Optimizer**: Used for optimization due to its efficiency and adaptive learning rate capabilities.
- **Learning Rate Scheduler (StepLR)**: Reduces the learning rate by a factor (`gamma`, e.g., 0.1) every few epochs (`step_size`, e.g., 7), allowing the model to converge more gracefully.

**Loss Function**:
- The Mean Squared Error (MSE) Loss measures the average squared difference between the predicted and actual values. It is particularly suitable for regression tasks like rating prediction.

```python
# Example: Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

### 5. Evaluation and Optimization

**Metrics Tracked**:
- **Training and Test Loss**
- **Accuracy**
- **Precision and Recall**

**Model Saving**:
- At the end of each epoch, the model's state, optimizer state, and metrics are saved to ensure that training can resume from the last checkpoint in case of interruptions.

```python
# Example: Saving model checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'accuracy': accuracy
}, path)
```

### 6. Recommendation Functions

**Popular Items**:
- Returns the most popular items based on average scores, serving as a fallback recommendation for new or inactive users.

**Content-Based Recommendations**:
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical form. Dimensionality reduction is then performed using Truncated SVD (Singular Value Decomposition). The Faiss library is used for efficient nearest-neighbor search, enabling quick retrieval of similar items.

```python
# Example: TF-IDF and Faiss for content-based filtering
tfidf_matrix = tfidf.fit_transform(df['combined_text'])
```

**Hybrid Recommendations**:
- Combines collaborative filtering and content-based filtering. If the user is new or has minimal interaction data, popular items are recommended. Otherwise, it first generates collaborative recommendations and then enhances them with content-based suggestions.

```python
# Example: Hybrid recommendation function
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
```

### 8. Integration of New Reviews

- New reviews are seamlessly integrated whenever available. This approach enables the model to stay updated with the latest user feedback and preferences, thereby improving the recommendation quality.

```python
# Example: Integrating new reviews
if os.path.exists(new_reviews_file):
    new_df = pd.read_csv(new_reviews_file)
    df = pd.concat([df, new_df], ignore_index=True)
```

## Continuous Evaluation
- **Logging**: Logs are stored in `app.log`.
- **Feedback**: User feedback is stored in `new_reviews.csv`.

## Scalability and Performance
- **Caching**: Uses Flask-Caching for low-latency responses.

## Evaluation and Optimization
- **Metrics**: Tracked and visualized over epochs for training/testing loss, accuracy, precision, and recall.

## Conclusion
This recommendation engine leverages collaborative filtering, content-based filtering, and a hybrid approach to provide personalized product recommendations. The training process is robust, incorporating techniques to prevent overfitting and ensuring scalability and performance. The model continuously evaluates and optimizes itself, making it a comprehensive solution for an e-commerce platform.
