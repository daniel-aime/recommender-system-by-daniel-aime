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

    @staticmethod
    def set_vector_uv(klass, matrix_user_item):
        """
        A short description.
            Calcule vecteur singulière gauche de A
        A bit longer description.
            U = AAt ===>At = Transposé de A
        Args:
            vector_u (DataFrame): vecteur singulière gauche de A sans encore remplit des valeurs singulière

        Returns:
            list: DataFrame pour le vecteur U et V

        Raises:
            Exception: description

        """

        matrix_user_item_transpose = matrix_user_item.transpose()
        vector_u = matrix_user_item.dot(matrix_user_item_transpose)
        vector_v = matrix_user_item_transpose.dot(matrix_user_item)

        return [vector_u, vector_v]

    @staticmethod
    def fill_sigma(factor = __name__.factor, eig_values = []) :
        sigma = np.zeros((factor, factor))

    @classmethod
    def set_svd():
        pass


user_item = pd.DataFrame([[3,2,2],[2,3,-2]],
                            index=[1,2], columns=[1,2,3])
test = [1,3,4]
print(sqrt(test))
