import numpy as np
import pandas as pd
#Fonction qui structure le données venant du BDD (Only user add rating on product)
def setInDataFrame(results:list):
    frame = {"user_id": [], "items_id": [], "rating": [] }
    for record in results.values():
        frame["user_id"].append(record[0]["user_id"])
        frame["items_id"].append(record[0]["items_id"])
        frame["rating"].append(record[0]["rating"])
    return frame

def setInDict(results):
    dict_items = {}
    for record in results:
        dict_items[record[0]["id"]] = record[0]["nom"]
    return dict_items

def setInList(results:list="items_id", item:str=None):
    """
        Description:
            Fonction qui prend un parametre d'un array dim 1*m
            et qui transforme et retourne une liste
        Parameters:
            results:List
            item: str
        Return:
         list

    """
    assert isinstance(results, list), "Le parametre results doit etre une liste."
    data = []
    for i, v in enumerate(results):
        data.append(v[0][item])
    return data

def encode_ids(data, to_ndarray=True):
    """
        Description:
            Stocker dans un dictionnaire tous les user_id unique et items_id afin de
            re-structurer le données avec un nouvel index et de colonnes
        Parameters:
            data(DataFrame) : user_item original
        return:
            data(DataFrame) : user_item encodé
            dict_users(dict) : user_id avec index
            dict_items(dict) : item_id avec index
            ratings(ndarray) : user_item encodé version numpy
    """
    data_encoded = data.copy()

    users = pd.DataFrame(data_encoded.index.unique(), columns=["users_id"])
    dict_users = users.to_dict()
    inv_dict_users = {v: k for k, v in dict_users['users_id'].items()}

    items = pd.DataFrame(data_encoded.columns.unique(), columns=["items_id"])
    dict_items = items.to_dict()
    inv_dict_items = {v: k for k, v in dict_items['items_id'].items()}

    data_encoded.index = data_encoded.index.map(inv_dict_users)
    data_encoded.columns = data_encoded.columns.map(inv_dict_items)
    # print(dict_items)
    # print(dict_users)
    # print(data_encoded.userId)
    if to_ndarray is True:
        return data_encoded.to_numpy()

    else: return data_encoded, dict_users, dict_items

def decode_ids(data_filled, data_original):
    """
        Description:
            Decoder le data encoder vers l'etat initial
        Parameters:
            data_filled(DataFrame) : user_item predicted
            data_original(DataFrame) : user_item original
        return:
            data_filled(DataFrame) : user_item decodé
    """
    data_encoded, dict_users, dict_items = encode_ids(data_original, to_ndarray=False)
    data_filled.rename(index=(dict_users["users_id"]), inplace=True)
    data_filled.rename(columns=(dict_items["items_id"]), inplace=True)

    return data_filled

def getNameProduct(items:dict, items_id):
    product = []
    for i in items_id:
        if i in items.keys():
            product.append(items[i])
    return product
