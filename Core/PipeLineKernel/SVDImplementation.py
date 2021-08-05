import sys
sys.path.append('..')
import math as mt
import numpy as np
import pandas as pd
import Logging as lg
import Standardization as sz
import metrics as mr
import matplotlib.pyplot as plt
from math import log
from abc import ABCMeta, abstractmethod
from DataPreprocessor import DataPreprocessor
import DataModele.DataFeaturing as df



class Data():

    def __init__(self, factor=None):
        data = DataPreprocessor()
        self.data_original_ui = data.set_dataset()
        self.user = data.user
        self.product_name = data.product_name
        self.user_item = data.ratings()
        self.ratings, self.test = data.train_test_split(test_value=False)
        self.factor = factor
        # self.all_user_id = list(self.user_item.index.values)
        # self.all_item__id = list(self.user_item.columns.values)

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


class Model(Data):
    def __init__(self,
                 factor = 40,
                 user_fact_reg = 0.0,
                 item_fact_reg = 0.0,
                 user_bias_reg = 0.0,
                 item_bias_reg = 0.0,
                 verbose = False):
        super().__init__()
        self.factor = self.low_rank_approximate() + 1
        self.n_users = self.ratings.shape[0]
        self.n_items = self.ratings.shape[1]
        self.user_fact_reg = user_fact_reg
        self.item_fact_reg = item_fact_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg
        self.verbose = verbose
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)

    def train(self, n_iter=15, learning_rate=0.001):
        """
        Entrainer le modele avec des donneés deja entraine dans partial_train
        jusqu'à n_iter itération
        """
        # Initialize randomly the lantent factors
        self.user_vect = np.random.normal(scale=1./self.factor,
                                          size=(self.n_users, self.factor))
        self.item_vect = np.random.normal(scale=1./self.factor,
                                          size=(self.n_items, self.factor))
        self.learning_rate = learning_rate

        # Initialize to zeros vector bias and globzl_bias using mean centered
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
        self.partial_train(n_iter)
        print("training...")


    def partial_train(self, n_iter):
        """
        Entrainer le modele jusqu'à n_iter iteration, On l'appeler plusieurs fois dans l'entrainement
        general du modele
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self.verbose:
                print(f"\t current iteration: {ctr}")

            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)
            self.sgd()
            ctr += 1


    @lg.log_timmer
    def sgd(self):
        for id_x in self.training_indices:
            u = self.sample_row[id_x]
            i = self.sample_col[id_x]
            prediction = self.predict(u, i)

            error = (self.ratings[u, i] - prediction) # error

            # Updating the users_vector and items_vector using partial derive
            self.user_bias[u] += self.learning_rate * (error - self.user_bias_reg * self.user_bias[u]) # Biais vecteur ligne utilisateur
            self.item_bias[i] += self.learning_rate * (error - self.item_bias_reg * self.item_bias[i]) # Biais vecteur colonne items

            # Updating the latent factors using partial derive
            self.user_vect[u, :] += self.learning_rate * \
                                    (error * self.item_vect[i, :] - self.user_fact_reg * self.user_vect[u, :]) # Matrice P

            self.item_vect[i, :] += self.learning_rate * \
                                    (error * self.user_vect[u, :] - self.item_fact_reg * self.item_vect[i, :])

    @lg.log_timmer
    def predict(self, u, i):
        """
        Predire le préference d'un utilisateurs d-un produit
        """
        prediction =self.global_bias + self.user_bias[u] + self.item_bias[i]
        # prediction += self.user_vect[u, :].dot(self.item_vect[i, :].T)
        prediction += np.dot(self.user_vect[u, :], self.item_vect[i, :].transpose()).astype('int32')

        return prediction

    @lg.log_timmer
    def predict_all(self):
        """Prediction à tout les utilisateurs et des produits"""
        prediction = np.zeros((self.user_vect.shape[0],
                               self.item_vect.shape[0]))

        for u in range(self.user_vect.shape[0]):
            for i in range(self.item_vect.shape[0]):
                prediction[u, i] = self.predict(u, i)

        return prediction

    def calculate_learning_curve(self,
                                iter_array,
                                learning_rate=0.01,
                                n_factor = 10,
                                returned=False):
        """"
        Surveiller le MSE et RMSE dans le periode d'entrainement du modele
        Parameters
        ----------
        iter_array : (list)
            Liste de nombres d'iteration dans le periode d'entrainement. ex: [1, 5, 10, 15]
        data_test : (2D ndarray)
            Le données de teste
        On créer aussi quatre nouveaux attribut de classe :
        train_rmse : (list)
            Valeurs de RMSE lors de la phase d'entrainement
        test_rmse : (list)
            Valeurs de RMSE lors de la phase de test
        train_mse : (list)
            Valeurs de MSE lors de la phase d'entrainement
        test_mse : (list)
            Valeurs de MSE lors de la phase de test
        """
        iter_array.sort()
        self.train_rmse = []
        self.test_rmse = []
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0

        for (i, n_iter) in enumerate(iter_array):
            if self.verbose :
                print(f"Iteration: {n_iter}")
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)
            predictions = self.predict_all()

            # Root Mean Square error
            self.train_rmse += [mr.rmse(self.ratings, predictions)]
            self.test_rmse += [mr.rmse(self.test, predictions)]

            # Mean Square error
            self.train_mse += [mr.mse(self.ratings, predictions)]
            self.test_mse += [mr.mse(self.test, predictions)]

            if self.verbose :
                print(f"Train rmse : {self.train_rmse[-1]}")
                print(f"Test rmse : {self.test_rmse[-1]}")
                print(f"Train mse : {self.train_mse[-1]}")
                print(f"Test mse : {self.test_mse[-1]}")
            iter_diff = n_iter

        if returned is True:
            return self.train_rmse, self.test_rmse, self.train_mse, self.test_mse

    @lg.log_timmer
    def optimize(self, iter_array,
                       n_factor=None, # Liste des facteur latente
                       regularization=None, # Liste de valeur de regularisation de nos vect bias
                       learning_rate=None, #  Liste de valeur des taux t'apprentissage
                       type_optimize="reg",
                       verbose=True): # Type of optimization
        """
        Description
        -----------
        Dans ce fonction, on va essayer d'optimiser tout nos paramètre

        Parameters
        ----------
        n_factor : (list)
            latent factor
        regularisation : (list)
            facteur de nos différent regurlarisation
        learning_rate : (list)
            Taux d'apprentissage alpha
        type_optimize : (str)
            Le type d'opimisation qu'on veut faire
            reg = regularization
            alpha = learning_rate

        Returns
        -------
        hyper_parameters : (dict)
        """
        if type_optimize not in ("reg", "alpha"):
            raise ValueError(f"{type_optimize} n'ont pas dans la ('reg', 'alpha')")

        # Dict of hyper arameters
        hyper_parameters = {}
        hyper_parameters['learning_rate'] = None
        hyper_parameters['regularization'] = (regularization[0], None)[type_optimize == "reg"]
        print(type(hyper_parameters["regularization"]))
        hyper_parameters['n_factor'] = (regularization[0], None)[type_optimize == "reg"]
        hyper_parameters['n_iter'] = 0
        hyper_parameters['train_rmse'] = np.inf
        hyper_parameters['test_rmse']  = np.inf
        hyper_parameters['train_mse'] = np.inf
        hyper_parameters['test_mse']  = np.inf


        if type_optimize == 'alpha':
            for rate in learning_rate:
                if verbose == True:
                    print(f"alpha : {rate}")
                self.calculate_learning_curve(iter_array, learning_rate=rate)
                rmse_index_min = np.argmin(self.test_rmse)
                mse_index_min = np.argmin(self.test_mse)

                # Search optimal parameter for rmse procedure
                if self.test_mse[mse_index_min] < hyper_parameters['test_mse']:
                    # Updata all new best parameter
                    hyper_parameters['learning_rate'] = rate
                    hyper_parameters['n_iter'] = iter_array[mse_index_min]
                    hyper_parameters['train_mse'] = self.train_mse[mse_index_min]
                    hyper_parameters['test_mse'] = self.test_mse[mse_index_min]

                # Serarch optimal parameter for mse procedure
                # if self.test_mse[mse_index_min] < hyper_parameters['test_mse']:
                #     #Update all new
        else:
            for fact in n_factor:
                for reg in regularization:
                    if verbose == True:
                        print(f"Lantente facteur: {fact} \t\t Regularisation: {reg} ")
                    self.user_fact_reg, self.item_fact_reg = reg
                    self.calculate_learning_curve(iter_array, learning_rate=0.001)
                    rmse_index_min = np.argmin(self.test_rmse)

                    # procedure to search the optimal parameter for rmse
                    if self.test_rmse[rmse_index_min] < hyper_parameters['test_rmse']:
                        # Update all new best parameter
                        hyper_parameters['n_factor'] = fact
                        hyper_parameters['regularization'] = reg
                        hyper_parameters['n_iter'] = iter_array[rmse_index_min]
                        hyper_parameters['train_rmse'] = self.train_rmse[rmse_index_min]
                        hyper_parameters['test_rmse'] = self.test_rmse[rmse_index_min]

        return hyper_parameters

    def plot_learning_curve(self,
                            iter_array,
                            train_rmse,
                            test_rmse,
                            train_mse,
                            test_mse):
        """
        Tracage de graphe qui montre la convergence de l'erreur de nos modele
        """

        # Graph Root Mean Square error
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(iter_array, train_rmse, label="Train RMSE") # Train courbe
        plt.plot(iter_array, test_rmse, label="Test RMSE") # Test courbe
        plt.title("Courbe de convergence de RMSE")
        plt.ylabel("RMSE")
        plt.xlabel("Nombre Iteration")
        plt.legend()
        # plt.show()
        # plt.savefig('Graph learning curve.png')

        # Graph Mean Square Error
        plt.subplot(1, 2, 2)
        plt.plot(iter_array, train_mse, c='blue', label="Train MSE")
        plt.plot(iter_array, test_mse, c='red', label="Test MSE")
        plt.title("Courbe de vonvergence de MSE")
        plt.ylabel("MSE")
        plt.xlabel("Nombre Iteration")
        plt.legend()
        plt.show()



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
        """
            1- Créer un bruit blanc de gaussien qui est centré zeros
            2- Trouver le mediane et le gamma
            2- Si la valeur d'intervale de confiance n'a pas encore respecter,
                on continue toujours la boucle SC = [80;90]%
        """
        beta = self.user_item.shape[1] / self.user_item.shape[0]
        tau = beta * np.median(self.values_singular)
        r = np.max(np.where(self.values_singular > tau))
        return r
        # predict = np.dot(np.dot(self.left_singular_vector, self.fill_sigma(nbr_factor=r)), self.right_singular_vector)







# test_2 = DataNoising()
