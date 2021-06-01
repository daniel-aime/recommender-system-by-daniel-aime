from .DatabaseAccess import Database
from .DataFeaturing import *


class ImportDatasetOnNeo4j:
    """
        Class singleton qui injecte les données dans le database neo4j
    """
    connection = Database.getConnection(uri="bolt://localhost:11003", user="test_user", password="root")
    __session__ = None
    __instance__ = None

    @staticmethod
    def getSingletonInstance():
        if ImportDatasetOnNeo4j.__instance__ is None:
            ImportDatasetOnNeo4j.__instance__ =  ImportDatasetOnNeo4j()
        return ImportDatasetOnNeo4j.__instance__



    @classmethod
    def createSession(cls):
        return cls.connection.session()


    @classmethod
    def getAllProduct(cls):
        query = """
                    MATCH (m:Movie)
                    RETURN {movieId:ID(m)}
                """
        cls.__session__ = cls.createSession()
        return setInList(results=cls.__session__.run(query).values(), item="movieId")


    @classmethod
    def getAllUser(cls):
        query = """
                    MATCH (n:Person)
                    return {userId:ID(n)}
                """
        cls.__session__ = cls.createSession()
        return setInList(results=cls.__session__.run(query).values(), item="userId")

    @classmethod
    def getUserRatingProduct(cls):
        query = """
                    MATCH (n:Person)-[r:REVIEWED]->(m:Movie) RETURN
                    {userId:ID(n), movieId:ID(m), rating:r.rating}
                """
        cls.__session__ = cls.createSession()
        return setInDataFrame(cls.__session__.run(query))
