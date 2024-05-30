import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
import joblib
from datetime import datetime
import os
from svd_tqdm import SVDTqdm

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load the reviews dataset
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

# Load feedback data
feedback_file_path = 'dataset/feedback.csv'
if os.path.exists(feedback_file_path):
    feedback_df = pd.read_csv(feedback_file_path)

    # Process feedback and adjust ratings
    for _, row in feedback_df.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']
        feedback = row['feedback']

        # Find the original rating
        original_rating = df[(df['UserId'] == user_id) & (df['ProductId'] == product_id)]['Score']

        if not original_rating.empty:
            if feedback == 'like':
                updated_rating = min(original_rating.values[0] + 1, 5)  # Increment rating
            elif feedback == 'dislike':
                updated_rating = max(original_rating.values[0] - 1, 1)  # Decrement rating

            # Update rating in the dataframe
            df.loc[(df['UserId'] == user_id) & (df['ProductId'] == product_id), 'Score'] = updated_rating
        else:
            if feedback == 'like':
                updated_rating = 5  # Assume a high rating for like
            elif feedback == 'dislike':
                updated_rating = 1  # Assume a low rating for dislike

            # Append the new feedback as a new rating
            new_row = pd.DataFrame({
                'UserId': [user_id],
                'ProductId': [product_id],
                'Score': [updated_rating],
                'HelpfulnessNumerator': [0],
                'HelpfulnessDenominator': [0],
                'Time': [datetime.now().timestamp()],
                'ProfileName': ['New_Feedback']
            })
            df = pd.concat([df, new_row], ignore_index=True)

# Using the surprise library for SVD model
# Creating a Reader object and specifying the rating scale
reader = Reader(rating_scale=(1, 5))

# Load data into surprise Dataset object
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_factors': [20, 50],
    'n_epochs': [20, 30],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}

gs = GridSearchCV(SVDTqdm, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Best model and parameters
best_model = gs.best_estimator['rmse']
best_params = gs.best_params['rmse']
print(f"Best RMSE: {gs.best_score['rmse']}")
print(f"Best Parameters: {best_params}")

# Train the best model
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Save the model
joblib.dump(best_model, 'models/svd_model.pkl')

# Save the product information dictionary
product_info = df[['ProductId', 'Summary']].drop_duplicates().set_index('ProductId').to_dict()
joblib.dump(product_info, 'models/product_info.pkl')

print("Model and product info saved successfully.")