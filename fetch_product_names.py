import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import concurrent.futures
import time
from filelock import FileLock

# Define the function to get product names
def get_product_name(asin):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }

    url = f"https://www.amazon.com/dp/{asin}"

    try:
        response = requests.get(url, headers=headers)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch product name for ASIN: {asin}. Error: {e}")
        return asin, None

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # The product title is usually within a tag with id 'productTitle'
        product_title_tag = soup.find(id='productTitle')

        if product_title_tag:
            product_title = product_title_tag.get_text().strip()
            return asin, product_title
        else:
            return asin, None
    else:
        # failed, print why
        print(f"Failed to fetch product name for ASIN: {asin}. Status code: {response.status_code}")
        return asin, None

# File paths
data_path = 'dataset/Reviews.csv'
output_path = 'dataset/ProductNames.csv'
lock_path = output_path + '.lock'

def fetch_names_concurrently():
    # Load the reviews dataset
    df = pd.read_csv(data_path)

    # Count unique product IDs
    num_unique_products = df['ProductId'].nunique()

    print(f"Number of unique products: {num_unique_products}")

    # Acquire lock to read/write ProductNames.csv
    lock = FileLock(lock_path)
    with lock:
        # Check if ProductNames.csv exists, and load it if it does
        if os.path.exists(output_path):
            product_names_df = pd.read_csv(output_path)
        else:
            product_names_df = pd.DataFrame(columns=['ProductId', 'ProductName'])

        # Create a set of already fetched ASINs to skip them
        fetched_asins = set(product_names_df['ProductId'])

    # Get list of ASINs to fetch
    asins_to_fetch = [asin for asin in df['ProductId'].unique() if asin not in fetched_asins]

    # Use ThreadPoolExecutor to fetch product names concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_asin = {executor.submit(get_product_name, asin): asin for asin in asins_to_fetch}

        for future in concurrent.futures.as_completed(future_to_asin):
            asin, product_name = future.result()
            if product_name:
                new_row = pd.DataFrame({'ProductId': [asin], 'ProductName': [product_name]})

                # Acquire lock to append new rows to ProductNames.csv
                with lock:
                    product_names_df = pd.concat([product_names_df, new_row], ignore_index=True)
                    product_names_df.to_csv(output_path, index=False)
                    print(f"Fetched: {asin} - {product_name}")

    print("Finished fetching product names.")

if __name__ == '__main__':

    fetch_names_concurrently()