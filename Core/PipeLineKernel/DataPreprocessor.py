import sys
sys.path.append('..')
from DataModele.DataSet import ImportDatasetOnNeo4j
import DataModele.DataFeaturing as df
import Standardization as sz
# mrz3x1

import numpy as np
import pandas as pd
np.random.seed(0)


class DataPreprocessor:
    __instance_dataset = ImportDatasetOnNeo4j.getSingletonInstance()
    def __init__(self):
        self.user_rating_product = DataPreprocessor.__instance_dataset.getUserRatingProduct()
        self.product_name = DataPreprocessor.__instance_dataset.getAllProductWithName()
        self.product = DataPreprocessor.__instance_dataset.getAllProduct()
        self.user = DataPreprocessor.__instance_dataset.getAllUser()

    def get_dataset(self):
        return self.dataset

    def get_unique_users(self):
        return np.unique(np.array([self.user]))

    def get_unique_items(self):
        return np.unique(np.array([self.product]))

    def set_dataset(self):
        """

        """
        items_unique = np.unique(np.array([self.product]))
        user_unique = np.unique(np.array([self.user]))

        tmp_user = None
        tmp_matrice = {users:{items:0 for items in items_unique} for users in user_unique}
        for item, user, rating in zip(self.user_rating_product['items_id'],
                         self.user_rating_product['user_id']
                        , self.user_rating_product['rating']):
                        if user not in tmp_matrice:
                            tmp_matrice[user] = {items:0 for items in items_unique}
                        tmp_matrice[user][item] = rating

        return pd.DataFrame([[val_rating for val_rating in val_user.values() ] for val_user in tmp_matrice.values()],
                                    index=user_unique, columns=items_unique)
    def ratings(self):
        return df.encode_ids(self.set_dataset(), to_ndarray=True)

    def train_test_split(self, test_value=True):
        """
        Validation croisé qui utilise la techinque de handout.
        Cross-Validation handout :
            -60% de données  pour le phase d'entrainement
            -40% reste destiner à la phase de test
        Données pris aléatoirement

        Args:
            test_value (bool): Valeur de test retourné

        Returns:
            test_value (ndarray): Valeur de test
            [train, test] (list[ndarray]) : Valeur d'entrainement et de test

        Raises:
            Exception: description

        """

        rating = df.encode_ids(self.set_dataset(), to_ndarray=True)
        train = rating.copy()
        test = np.zeros(rating.shape)
        for user in range(rating.shape[0]):
            test_ratings = np.random.choice(rating[user, :].nonzero()[0],
                                            size=3,
                                            replace=True)

            train[user, test_ratings] = 0.
            test[user, test_ratings] = rating[user, test_ratings]
        assert(np.all((train * test)) == 0)
        if test_value:
            return test
        else: return train, test

    @classmethod
    def standardize(klass):
        """
        Calcul ecart du base_predict et remplir le zeros avant
        Sc = S - Sb
        """
        matrix_user_item = klass.set_dataset()
        matrix_standard = sz.base_predict(matrix_user_item.copy(deep=True))
        cols = matrix_user_item.columns.values
        indx = matrix_user_item.index.values
        std_standard = pd.DataFrame(np.zeros((len(indx),len(cols))),index=indx,columns=cols)
        for ind in matrix_user_item.index.values:
            for col in matrix_user_item.columns.values:
                if matrix_user_item.loc[ind, col] == 0:
                    continue
                else:
                    std_standard.loc[ind, col] = matrix_user_item.loc[ind, col] - matrix_standard.loc[ind, col]
        return std_standard


    def items_similarity(self, type_correlation="pearson"):
        """
        Calcul simulitude a chaque items
        Args:
            std_standard (serie.pandas): Matrice normalisé par Sc = S - Sb
            returns (str): [sum, None]
            correlation (str): type de correlation utilisé dans le similutide des items

        Returns:
            items_similarity: DataFrame
                Dictionnaire qui stock chaque similitude d'items par rapport aux autres items
        Raises:
            ValueError: Le type de correlation ne figure pas dans le liste de type pris en charge de la fonction
        """
        list_value = ['pearson', 'spearman', 'kendall']
        if type_correlation not in list_value:
            raise ValueError(f"Type correlation has not in{list_value}")
        matrix_standard = self.standardize()
        return matrix_standard






# movie_rating = pd.DataFrame([[0,2,0,1,7],[1,4,0,0,0],[1,0,0,2,0],[3,1,0,0,0]],
#                             index=[1,2,3,4], columns=[1,2,3,4,5])
# dp = DataPreprocessor()
# # # train, test = dp.train_test_split()
# print(dp.product_name)
# # # print(test)
