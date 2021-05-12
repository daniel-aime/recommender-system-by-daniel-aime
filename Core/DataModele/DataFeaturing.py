#Fonction qui structure le données venant du BDD (Only user add rating on product)
def setInDataFrame(results:list):
    frame = {"user_id": [], "movie_id": [], "rating": [] }
    for record in results.values():
        frame["user_id"].append(record[0]["userId"])
        frame["movie_id"].append(record[0]["movieId"])
        frame["rating"].append(record[0]["rating"])
    return frame

def setInList(results:list):
    """
        Description:
            Fonction qui prend un parametre d'un array dim 1*m
            et qui transforme et retourne une liste
        Parameters:
            results:List
        Return:
         list

    """
    return [i for i,v in enumerate(results.values())]
