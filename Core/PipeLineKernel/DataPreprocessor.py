import numpy as np
import pandas as pd
import sys

sys.path.append('..')
from DataModele.DataSet import ImportDatasetOnNeo4j
class DataPreprocessor:
    __instance_dataset = ImportDatasetOnNeo4j.getSingletonInstance()
    def __init__(self):
        self.user_rating_product = DataPreprocessor.__instance_dataset.getUserRatingProduct()
        self.product_no_rating = DataPreprocessor.__instance_dataset.getAllProductNoRating()
    def get_dataset(self):
        return self.dataset

    def set_dataset(self):
        self.dataset = dataset

test = DataPreprocessor()
print(test.user_rating_product)
