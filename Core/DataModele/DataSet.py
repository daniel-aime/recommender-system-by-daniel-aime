from DatabaseAccess import Database
from DataFeaturing import *


class ImportDatasetOnNeo4j:
    connection = Database.getConnection(uri="bolt://localhost:7687", user="test_user", password="root")
    __session__ = None
    def __init__(self):
        pass

    @classmethod
    def createSession(cls):
        return cls.connection.session()


    @classmethod
    def getAllProductNoRating(cls):
        query = """
                    MATCH (m:Movie) WHERE NOT (m)<-[:REVIEWED]-()
                    RETURN ID(m)
                """
        cls.__session__ = cls.createSession()
        return setInDict(cls.__session__.run(query))


    @classmethod
    def getUserNoRatingProduct(cls):
        query = """
                    MATCH (n:Person) WHERE NOT (n)-[:REVIEWED]->(:Movie)
                    return {userId:ID(n)}
                """
        cls.__session__ = cls.createSession()
        return cls.__session__.run(query)

    @classmethod
    def getUserRatingProduct(cls):
        query = """
                    MATCH (n:Person)-[r:REVIEWED]->(m:Movie) RETURN
                    {userId:ID(n), movieId:ID(m), rating:r.rating}
                """
        cls.__session__ = cls.createSession()
        return setInDataFrame(cls.__session__.run(query))

test = ImportDatasetOnNeo4j.getAllProductNoRating()
print(test)
