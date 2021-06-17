import math as mt
import numpy as np
import pandas as pd
import Standardization as sz
import matplotlib.pyplot as plt
from math import log
from abc import ABCMeta, abstractmethod

class Data():

    def __init__(self, factor=2):
        self.user_item = pd.DataFrame([[1,1,1,0,0,1,3],[3,3,3,0,0,0,1], [1,4,3,0,0,2,5], [0,0,2,0,0,2,1], [0,0,1,0,0,5,0], [5,5,5,0,0,1,3], [0,2,0,4,4,5,2], [0,0,0,5,5,2,0], [0,1,0,2,2,1,1]],
                                    index=[1,2,3,4,5,6,7,8,9], columns=[1,2,3,4,5,6,7])
        self.factor = factor
        self.all_user_id = list(self.user_item.index.values)
        self.all_item__id = list(self.user_item.columns.values)

    def set_vector_u(self):
        """
        A short description.
            Calcule vecteur singulière gauche de A
        A bit longer description.
            U = AAt ===> At = Transposé de A
        Args:
            vector_u (DataFrame): vecteur singulière gauche de A sans encore remplit des valeurs singulière

        Returns:
            list: DataFrame pour le vecteur U et V

        Raises:
            Exception: description

        """
        user_item_transpose = self.user_item.transpose()
        vector_u = self.user_item.dot(user_item_transpose)
        w, u = np.linalg.eig(vector_u.astype('float32'))
        return w,u

    def set_vector_v(self) :
        user_item_transpose = self.user_item.transpose()
        vector_v = user_item_transpose.dot(self.user_item)
        w, v = np.linalg.eig(abs(vector_v.astype('float32')))
        return w,v


    @property
    def values_singular(self, nbr_round= 3):
        w_1 = [eig_values for eig_values in self.set_vector_u()[0]]
        w_2 = [eig_values for eig_values in self.set_vector_v()[0]]
        w_1.extend(w_2)
        w_rounded_value = [round(abs(values), 10) for values in w_1]
        del w_1, w_2
        for i,j in enumerate(w_rounded_value):
            if w_rounded_value.count(j) > 1:
                w_rounded_value.pop(i)
        w_rounded_value.sort(reverse=True)

        return w_rounded_value

    @property
    def left_singular_vector(self):
        return self.set_vector_u()[1][:, :self.factor].astype('float32')

    @property
    def right_singular_vector(self):
        return self.set_vector_v()[1].transpose()[:self.factor, :]


    def fill_sigma(self, nbr_factor=None) :
        if nbr_factor == "all":
            singular_values = np.sqrt(self.values_singular)
            return singular_values
        else:
            if nbr_factor is None:
                nbr_factor = self.factor
            else: nbr_factor = nbr_factor

            singular_values = np.sqrt(self.values_singular[:nbr_factor])
            sigma = np.zeros((nbr_factor, nbr_factor))
            singular_values = np.sqrt(self.values_singular[:nbr_factor])
            np.fill_diagonal(sigma, singular_values )
            return sigma



class DatasetFactory(object):
    @classmethod
    def newData(cls, name_object):
        return eval(name_object)()

"""
SVD de A = U * Sigma * Vtranspose
Etape à suivre pour le SVD:
    1: Il faut trouver le vecteur U = AtA respectivement pour V = AAt
    2: Trouver la valeur singulière de A afin de construire Sigma
    3: Trouver les vecteurs singulières gauche et droite du matrice A


"""
class DataNoising(Data):
    def __init__(self, epsilon=1):
        super().__init__()
        self.epsilon = epsilon
        self.data_noise = None
    @property
    def data_noise(self):
        return self.data_noise

    @data_noise.setter
    def data_noise(self, epsilon):
        user_item = self.user_item
        self.user_item = user_item + epsilon * np.random.randn(*user_item.shape)

class DataClean(Data):
    def __init__(self, factor=2):
        super().__init__()


    @staticmethod
    def find_users_rated_items(matrix_user_item, sim_users_uk, item_id):
        user_item = matrix_user_item.loc[:, item_id]
        user_item = list(user_item[user_item.loc[:] != 0].index.values) #= [1,3,4] = [3,5,2,1]
        sim = sim_users_uk[user_item[0]]
        for i in user_item:
            if i not in sim_users_uk.keys():
                sim = min(sim_users_uk, key=sim_users_uk.get)
            elif sim > sim_users_uk[i]:
                sim = i
        return sim

    def item_recommended(self, user_id):
        user_similaire = sz.sim_users_uk(self.right_singular_vector, self.all_user_id, user_id)
        items_fast_recommended = []
        for i in user_similaire:
            for note_user_y, note_user_x, item in zip(list(self.user_item.loc[i, :]), list(self.user_item.loc[user_id, :]), list(self.user_item.columns.values)):
                if note_user_y != 0 and note_user_x == 0:
                    if item in items_fast_recommended:
                        continue
                    else:
                        items_fast_recommended.append(item)
                # print(f"user = {i} note item : {item} = {note} et ")
        return items_fast_recommended

    @staticmethod
    def compute_svd(right_vector, sigma, transpose_left_vector):
        """
        Calculer la decomposition du matrice user_item.
        Formule:
            svd(A) = sig1*U_1*V_1T + sig2*U_2*V_2T +....+ sigi*U_i*V_iT

                u1    1 3
                u2    2 0
                u3    3 0
                u4    4 5

                u1    1 3
                u2    2 5
                u3    3 1
                u4    4 5
        """

    @staticmethod
    def sum_pow_list(list):
        return sum([x**2 for x in list])

    @staticmethod
    def frobenuis_error(mat_init, mat_approx):
        error = 0
        for i in mat_approx.index.values:
            for v in mat_approx.columns.values:
                error += (mat_init.loc[i, v] - mat_approx.loc[i, v])**2
        return mt.sqrt(error)

    def low_rank_approximate(self):
        n_factor = self.factor #nbr facteur
        alpha = 2 #learning rate
        n_epoch = 10 #number iteration of SGD procedure

        """
            1- Créer un bruit blanc de gaussien qui est centré zeros
            2- Trouver le mediane et le gamma
            2- Si la valeur d'intervale de confiance n'a pas encore respecter,
                on continue toujours la boucle SC = [80;90]%
        """
        sigma = self.fill_sigma(nbr_factor="all")
        med_sigma = np.median(sigma)
        return med_sigma, sigma





test_2 = DataNoising()
test_2.data_noise = 2
print(test_2.data_noise)
