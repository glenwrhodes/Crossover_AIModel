from surprise import SVD
from tqdm import tqdm

class SVDTqdm(SVD):
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, **kwargs):
        super().__init__(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, **kwargs)
        self.lr_all = lr_all
        self.reg_all = reg_all

    def fit(self, trainset):
        """Fit the model to the training set with a progress bar."""
        self.trainset = trainset

        # Initialize biases and factors
        self.bu = [0.0] * trainset.n_users
        self.bi = [0.0] * trainset.n_items
        self.pu = [[0.0] * self.n_factors for _ in range(trainset.n_users)]
        self.qi = [[0.0] * self.n_factors for _ in range(trainset.n_items)]

        # Place the tqdm progress bar around the existing training loop
        for current_epoch in tqdm(range(self.n_epochs), desc="Training SVD"):
            super().fit(trainset)

        return self
    