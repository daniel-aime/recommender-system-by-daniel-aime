#Fonction qui structure le données venant du BDD (Only user add rating on product)
def setInDataFrame(results:list):
    frame = {"user_id": [], "movie_id": [], "rating": [] }
    for record in results.values():
        frame["user_id"].append(record[0]["userId"])
        frame["movie_id"].append(record[0]["movieId"])
        frame["rating"].append(record[0]["rating"])
    return frame

def setInList(results:list="movieId", item:str=None):
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
