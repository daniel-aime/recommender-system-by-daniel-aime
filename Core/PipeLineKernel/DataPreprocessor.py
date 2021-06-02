import sys
sys.path.append('..')
from DataModele.DataSet import ImportDatasetOnNeo4j
import Standardization as sz
# mrz3x1

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

        """
        items_unique = np.unique(np.array([self.product]))
        user_unique = np.unique(np.array([self.user]))

        tmp_user = None
        tmp_matrice = {users:{items:0 for items in items_unique} for users in user_unique}
        for item, user, rating in zip(self.user_rating_product['movie_id'],
                         self.user_rating_product['user_id']
                        , self.user_rating_product['rating']):
                        if user not in tmp_matrice:
                            tmp_matrice[user] = {items:0 for items in items_unique}
                        tmp_matrice[user][item] = rating

        return pd.DataFrame([[val_rating for val_rating in val_user.values() ] for val_user in tmp_matrice.values()],
                                    index=user_unique, columns=items_unique)

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


        



movie_rating = pd.DataFrame([[0,2,0,1,7],[1,4,0,0,0],[1,0,0,2,0],[3,1,0,0,0]],
                            index=[1,2,3,4], columns=[1,2,3,4,5])
test = DataPreprocessor.standardize()
print(test)
