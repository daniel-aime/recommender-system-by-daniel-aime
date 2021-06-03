import numpy as np
import pandas as pd

"""
SVD de A = U * Sigma * Vtranspose
Etape à suivre pour le SVD:
    1: Il faut trouver le vecteur U = AtA respectivement pour V = AAt
    2: Trouver la valeur singulière de A afin de construire Sigma
    3: Trouver les vecteurs singulières gauche et droite du matrice A


"""
class SVD():
    def __init__(self, factor=2):
        self.factor = factor
        self.user_item = pd.DataFrame([[1,1,1,0,0],[3,3,3,0,0], [4,4,4,0,0], [5,5,5,0,0], [0,2,0,4,4], [0,0,0,5,5], [0,1,0,2,2]],
                                    index=[1,2,3,4,5,6,7], columns=[1,2,3,4,5])

    def set_vector_u(self):
        """
        A short description.
            Calcule vecteur singulière gauche de A
        A bit longer description.
            U = AtA ===> At = Transposé de A
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
        w, v = np.linalg.eig(vector_v.astype('float32'))
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
    def right_singular_vector(self):
        return self.set_vector_u()[1][:, :self.factor]

    @property
    def left_singular_vector(self):
        return self.set_vector_v()[1].transpose()[:self.factor, :]


    def fill_sigma(self, nbr_factors = 3) :
        sigma = np.zeros((nbr_factors, nbr_factors))
        np.fill_diagonal(sigma, self.values_singular[:nbr_factors])
        return sigma

    @classmethod
    def set_svd():
        pass



w = SVD()
# dd = np.array([[1,2,1,4], [3,1,4,1]])
# dd = dd[:, :2]
print(w.right_singular_vector)
