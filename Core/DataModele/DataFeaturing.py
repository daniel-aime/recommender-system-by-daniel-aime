def setInDataFrame(results):
    frame = {"user_id": [], "movie_id": [], "rating": [] }
    for record in results.values():
        frame["user_id"].append(record[0]["userId"])
        frame["movie_id"].append(record[0]["movieId"])
        frame["rating"].append(record[0]["rating"])
    return frame
