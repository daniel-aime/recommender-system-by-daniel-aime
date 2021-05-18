import sys
sys.path.append('..')
from DataModele.DataSet import ImportDatasetOnNeo4j
from surprise import Reader as rd
from surprise import Dataset as dt


import numpy as np
import pandas as pd


class DataPreprocessor:
    __instance_dataset = ImportDatasetOnNeo4j.getSingletonInstance()
    def __init__(self):
        self.user_rating_product = DataPreprocessor.__instance_dataset.getUserRatingProduct()
        self.product = DataPreprocessor.__instance_dataset.getAllProduct()
        self.user = DataPreprocessor.__instance_dataset.getAllUser()


    def get_dataset(self):
        return self.dataset

    def set_dataset(self):
        """
                prod_1 prod_2 prod_3 prod_4 prod_5
        user_1  1      0      4      0      0
        user_2  0      0      0      1      0
        user_3  0      0      6      1      0
        user_4  0      1      1      0      5
        user_5  0      4      0      0      1

        """
        movie_unique = np.unique(np.array([self.product]))
        user_unique = np.unique(np.array([self.user]))

        tmp_user = None
        tmp_matrice = {users:{movies:0 for movies in movie_unique} for users in user_unique}
        for movie, user, rating in zip(self.user_rating_product['movie_id'],
                         self.user_rating_product['user_id']
                        , self.user_rating_product['rating']):
                        if user not in tmp_matrice:
                            tmp_matrice[user] = {movies:0 for movies in movie_unique}
                        tmp_matrice[user][movie] = rating
        """
            df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],

                  index=[4, 5, 6], columns=['A', 'B', 'C'])
            >>output
                A   B   C
            4   0   2   3
            5   0   4   1
            6  10  20  30

        """
        movie_rating = pd.DataFrame([[val_rating for val_rating in val_user.values() ] for val_user in tmp_matrice.values()],
                                    index=user_unique, columns=movie_unique)

        print(movie_rating)
        # print(self.user_rating_product)
        # print(tmp_matrice)





        # print(user_rate_movie)
        # print(movie)


    def setInDataFrame(self):
        print(self.user_rating_product['user_id'].sort())
test = DataPreprocessor()
test.set_dataset()
