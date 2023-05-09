import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


class KohonenNetwork:
    def __init__(self, input_shape, output_shape, learning_rate=0.1):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.weights = np.random.random((output_shape[0], output_shape[1], input_shape))
        self.neighborhood_radius = max(output_shape) / 2

    def get_winner(self, input_vector):
        distances = np.sum(np.square(self.weights - input_vector), axis=-1)
        winner = np.unravel_index(np.argmin(distances), self.output_shape)
        return winner

    def update_weights(self, input_vector, winner):
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                dist = np.sum(np.square(np.array([i, j]) - np.array(winner)))
                if dist <= self.neighborhood_radius:
                    lr = self.learning_rate * np.exp(-dist / (2 * self.neighborhood_radius))
                    self.weights[i, j] += lr * (input_vector - self.weights[i, j])

    def train(self, data, epochs):
        for epoch in range(epochs):
            np.random.shuffle(data)
            for input_vector in data:
                winner = self.get_winner(input_vector)
                self.update_weights(input_vector, winner)

def normalize_data(data):
    normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return normalized_data

class SOM:
    def __init__(self, n_rows, n_cols, n_features, lr=0.1, sigma=None, n_iterations=100):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_features = n_features
        self.lr = lr
        self.sigma = sigma if sigma is not None else max(n_rows, n_cols) / 2
        self.n_iterations = n_iterations
        self.weights = np.random.randn(n_rows, n_cols, n_features)

    def _find_bmu(self, x):
        # print(x.shape)
        x = np.reshape(x, (1, 1, -1))  # Reshape input vector to (1, 1, 15)

        distances = np.linalg.norm(self.weights - x, axis=-1)
        bmu_idx = np.argmin(distances)
        bmu_row, bmu_col = np.unravel_index(bmu_idx, (self.n_rows, self.n_cols))
        return bmu_row, bmu_col

    def train(self, data):
        for iteration in range(self.n_iterations):
            for x in data:
                bmu_row, bmu_col = self._find_bmu(x)
                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        weight = self.weights[r, c, :]
                        distance_to_bmu = np.linalg.norm([r - bmu_row, c - bmu_col])
                        neighbourhood = np.exp(-(distance_to_bmu ** 2) / (2 * (self.sigma ** 2)))
                        self.weights[r, c, :] = weight + neighbourhood * self.lr * (x - weight)

    def predict(self, x):
        bmu_row, bmu_col = self._find_bmu(x)
        return bmu_row, bmu_col



def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    numeric_cols = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                    'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
                    'citympg', 'highwaympg', 'price']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)
    data = data[numeric_cols].values
    data = normalize_data(data)  # normalize data between 0 and 1
    return data

def plot_maps(map):
    n_features = map.weights.shape[2]
    fig, axs = plt.subplots(nrows=1, ncols=n_features, figsize=(10, 2))
    for i in range(n_features):
        axs[i].imshow(np.flipud(map.weights[:,:,i]), cmap='jet')
        axs[i].set_title(f'Feature {i+1}')
        axs[i].axis('off')
    plt.show()


if __name__ == '__main__':
    data = load_csv_data('top20.csv')
    map = SOM(n_rows=10, n_cols=10, n_features=15, lr=0.1, sigma=None, n_iterations=100)
    map.train(data)
    print(map.weights)

    plot_maps(map)
