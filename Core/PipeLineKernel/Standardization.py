
import pandas as pd
import numpy as np
import math as mt
from itertools import product


def mean_rating_users(matrix_user_item):
    """
    Description:
        Creation une vecteur biais pour le note de chaque utilisateur par item.

    S(Ui, *).

    Args:
        matrice_user_item (serie.pandas): Toute les notes distribué par les utilisateurs

    Returns:
        list: Liste des vecteur colonne biais des utilisateurs

    Raises:
        Exception: description

    """

    vect_bais = []
    for v in matrix_user_item.index:
        nbr_rating_for_user_i = abs(len(matrix_user_item.loc[v, :]) - matrix_user_item.loc[v, :].value_counts()[0])
        if nbr_rating_for_user_i != 0:
            #Somme de note attribué v divisé par nbr d'attribution par l'user vi
            biais_ui = matrix_user_item.loc[v, :].sum()/nbr_rating_for_user_i
        else:
            biais_ui = 0
        vect_bais.append(biais_ui)
    return vect_bais


def mean_rating_items(matrix_user_item):
    """
    Creation une vecteur biais pour le note de chaque utilisateur par item.

    S(*, Ij). Ij ==> Item de i=1 à nombre total des article

    Args:
        matrice_user_item (serie.pandas): Toute les notes distribué par les utilisateurs

    Returns:
        list: Liste des vecteur colonne biais des utilisateurs

    Raises:
        Exception: description

    """
    vect_bais = []
    # print(movie_rating)
    mean_note_biais = 0
    for v in matrix_user_item.columns:
        nbr_rating_item_i = abs(len(matrix_user_item.loc[:, v]) - matrix_user_item.loc[:, v].value_counts()[0])
        if nbr_rating_item_i != 0:
            biais_iu = matrix_user_item.loc[:, v].sum() / nbr_rating_item_i
        else:
            biais_iu = 0
        vect_bais.append(biais_iu)
        mean_note_biais += biais_iu
    vect_bais.append(mean_note_biais/len(vect_bais))
    return vect_bais
##Test
def biais_vector(mean_rating_users:list=None, mean_rating_items:list=None, matrix_user_item=None):
    """
    Generation des biais du matrice_user_item
    """
    if (mean_rating_items or mean_rating_users or matrix_user_item) is None:
        raise AttributeError("One parameter missing in VectorBiais")
    else:
        matrix_user_item['mean_rating_users'] = pd.Series(mean_rating_users, index=matrix_user_item.index)
        matrix_user_item.loc['mean_rating_items'] = mean_rating_items
    return matrix_user_item


def global_mean_rating(matrix_user_item=None):
    return mean_rating_items(matrix_user_item)[-1]

def base_predict(matrix_user_item):
    """
    Prediction de base pour m*n et completion du matrice
    S(Ui,Aj) = Sim(*,*)+(Sim(Ui,*) - Sim(*,*))  + (Sim(*, Aj) - Sim(*,*))
    """
    sim_user_items = mean_rating_users(matrix_user_item)
    sim_items_user = mean_rating_items(matrix_user_item)
    mean_rating = global_mean_rating(matrix_user_item)
    mat_predict = matrix_user_item
    for i,indx in enumerate(matrix_user_item.index.values):
        for c,col in enumerate(matrix_user_item.columns.values):
            mat_predict.loc[indx, col] = mean_rating + (sim_user_items[i] - mean_rating) + (sim_items_user[c] - mean_rating)
    return mat_predict

def sim_items(std_standard, type_correlation="pearson"):
    """
    Calcul simulitude a chaque items
    Args:
        std_standard (serie.pandas): Matrice normalisé par Sc = S - Sb
        returns (str): [sum, None]
        correlation (str): type de correlation utilisé dans le similutide des items

    Returns:
        DataFrame:Dictionnaire qui stock chaque similitude d'items par rapport aux autres items
    Raises:
        Exception: description
    """
    list_value = ['pearson', 'spearman', 'kendall']
    if type_correlation not in list_value:
        raise ValueError(f"Type correlation has not in{list_value}")

    return (1 + std_standard.corr(method=type_correlation)) / 2

def sim_users_uk(right_vector, all_user_id, user_id):
    """
    u1    1 3
    u2    2 5
    u3    3 1
    u4    4 5
    """
    # print(right_vector)
    right_singular_vector = pd.DataFrame(right_vector, index=all_user_id)
    vector_user = list(right_singular_vector.loc[user_id])
    sim_user_x_to_y = {}
    for i, value in enumerate(right_singular_vector.index.values):
        if value != user_id:
            x_dot_y = 0
            scalaire_x = 0
            scalaire_y = 0
            for values_concept_user, other_value_concept in zip(vector_user, right_singular_vector.loc[value]):
                x_dot_y  += values_concept_user * other_value_concept
                scalaire_x += values_concept_user**2
                scalaire_y += other_value_concept**2
            sim_user_x_to_y[value] = x_dot_y / (mt.sqrt(scalaire_x) * mt.sqrt(scalaire_y))
            del x_dot_y, scalaire_x, scalaire_y
    user_sim = []
    mean_value = (sum(sim_user_x_to_y.values()) / len(sim_user_x_to_y))
    for i, v in sim_user_x_to_y.items():
        if v < mean_value:
            user_sim.append(i)
    # print(sim_user_x_to_y)
    return user_sim

def find_users_rated_items(matrix_user_item, sim_users_uk:dict, item_id):
    user_item = matrix_user_item.loc[:, item_id]
    user_item = user_item[user_item.loc[:] != 0]
    list_user = {2:0.0001, 3:0.01, 4:0.4, 5:0.001}
    sim = min(sim_users_uk, key=sim_users_uk.get)
    return sim



# test = np.array([[5,0,0,4],[0,4,3,0],[4,1,2,0]])
# movie_rating = pd.DataFrame(test,
#                             index=[1,2,3], columns=[1,2,3,4])
# print(find_users_rated_items(movie_rating,3))
