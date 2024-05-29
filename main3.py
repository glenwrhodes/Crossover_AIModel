import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Load the dataset
data_path = 'dataset/Reviews.csv'
df = pd.read_csv(data_path)

# Display the first few rows and data info
print(df.head())

# Handle missing values using forward fill
df.ffill(inplace=True)

# Normalize/clean data by removing duplicates
df.drop_duplicates(inplace=True)

# Downcast numeric types to save memory
df['HelpfulnessNumerator'] = pd.to_numeric(df['HelpfulnessNumerator'], downcast='integer')
df['HelpfulnessDenominator'] = pd.to_numeric(df['HelpfulnessDenominator'], downcast='integer')
df['Score'] = pd.to_numeric(df['Score'], downcast='integer')
df['Time'] = pd.to_numeric(df['Time'], downcast='integer')

# Using the surprise library for SVD model
# Creating a Reader object and specifying the rating scale
reader = Reader(rating_scale=(1, 5))

# Load data into surprise Dataset object
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)

# Split into training and testing sets within surprise
trainset, testset = surprise_train_test_split(data, test_size=0.2)

# Initialize and train the SVD model with custom parameters
model = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# Make predictions on the testset
predictions = model.test(testset)

# Evaluate the model (we can use RMSE as one of the evaluation metrics)
accuracy.rmse(predictions)

# Create a dictionary to map product IDs to summaries
product_info = df[['ProductId', 'Summary']].drop_duplicates().set_index('ProductId').to_dict()['Summary']

# Function to generate top N recommendations for a user
def get_top_n_recommendations(user_id, model, df, product_info, n=10):
    """
    Generate top N product recommendations for a given user.

    Parameters:
    user_id (str): The ID of the user to recommend products to.
    model (surprise.AlgoBase): The trained recommendation model.
    df (pd.DataFrame): The original dataframe containing user-product interactions.
    product_info (dict): A dictionary mapping ProductId to product summaries.
    n (int): The number of recommendations to generate.

    Returns:
    list: A list of tuples, each containing (product_id, predicted_rating, summary).
    """
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

# Example usage
user_id = 'A3SGXH7AUHU8GW'  # Replace with an actual user ID from your dataset
top_recommendations = get_top_n_recommendations(user_id, model, df, product_info, n=10)

print("Top 10 product recommendations for user {}: ".format(user_id))
for product_id, predicted_rating, summary in top_recommendations:
    print("Product ID: {}, Predicted Rating: {:.2f}, Summary: {}".format(product_id, predicted_rating, summary))

    