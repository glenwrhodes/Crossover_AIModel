import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy
import joblib
from svd_tqdm import SVDTqdm

# Load the dataset
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

# Using the surprise library for SVD model
# Creating a Reader object and specifying the rating scale
reader = Reader(rating_scale=(1, 5))

# Load data into surprise Dataset object
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)

# Split into training and testing sets within surprise
trainset, testset = surprise_train_test_split(data, test_size=0.2)

# Initialize and train the SVD model with custom parameters and progress bar
model = SVDTqdm(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# Evaluate the model
predictions = model.test(testset)
accuracy.rmse(predictions)

# Save the model
joblib.dump(model, 'models/svd_model.pkl')

# Save the product information dictionary
product_info = df[['ProductId', 'Summary']].drop_duplicates().set_index('ProductId').to_dict()['Summary']
joblib.dump(product_info, 'models/product_info.pkl')

print("Model and product info saved successfully.")
