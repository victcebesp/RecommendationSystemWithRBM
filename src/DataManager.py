import numpy as np
import pandas as pd
import torch
from src.DatasetType import DatasetType

class Datamanager:

    def __init__(self, training_set_path, test_set_path):
        self.training_set = self.transform_to_tensor(training_set_path, set_type=DatasetType.TRAIN)
        self.test_set = self.transform_to_tensor(test_set_path, set_type=DatasetType.TEST)

        self.number_of_users = int(max(max(self.training_set[:, 0]), max(self.test_set[:, 0])))
        self.number_of_movies = int(max(max(self.training_set[:, 1]), max(self.test_set[:, 1])))

    def convert(self, dataset):
        converted_dataset = []
        for user_id in range(1, self.number_of_users + 1):
            movies_id = dataset[:, 1][dataset[:, 0] == user_id]
            ratings_id = dataset[:, 2][dataset[:, 0] == user_id]
            ratings = np.zeros(self.number_of_movies)
            ratings[movies_id - 1] = ratings_id
            converted_dataset.append(list(ratings))
        return converted_dataset

    def transform_sets_to_binary(self, dataset):
        transformed_dataset = dataset
        transformed_dataset[transformed_dataset == 0] = -1
        transformed_dataset[transformed_dataset == 1] = 0
        transformed_dataset[transformed_dataset == 2] = 0
        transformed_dataset[transformed_dataset >= 3] = 1
        return transformed_dataset

    def transform_to_tensor(self, set_path, set_type):
        dataset = pd.read_csv(set_path, delimiter='\t').values
        converted_set = self.convert(dataset)
        if set_type == DatasetType.TRAIN:
            tensor_set = torch.FloatTensor(converted_set)
            return self.transform_sets_to_binary(tensor_set)
        else:
            tensor_set = torch.FloatTensor(converted_set)
            return self.transform_sets_to_binary(tensor_set)
