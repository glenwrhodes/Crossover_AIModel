import pandas as pd
from torch.utils.data import Dataset, DataLoader

class AmazonReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.user_to_idx = {user: idx for idx, user in enumerate(self.data['UserId'].unique())}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.data['ProductId'].unique())}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        self.user_ids = self.data['UserId'].map(self.user_to_idx).astype('int')
        self.item_ids = self.data['ProductId'].map(self.item_to_idx).astype('int')
        self.scores = self.data['Score']
        self.num_users = len(self.user_to_idx)
        self.num_items = len(self.item_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        score = self.scores[idx]
        return user_id, item_id, score

def get_data_loader(data, batch_size):
    dataset = AmazonReviewDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.num_users, dataset.num_items, dataset.user_to_idx, dataset.idx_to_user, dataset.item_to_idx, dataset.idx_to_item