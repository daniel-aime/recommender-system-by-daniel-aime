from .DatabaseAccess import Database
from .DataFeaturing import *


class ImportDatasetOnNeo4j:
    """
        Class singleton qui injecte les données dans le database neo4j
    """
    connection = Database.getConnection(uri="bolt://localhost:11006", user="test_user", password="root")
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
                    MATCH (m:Produit)
                    RETURN {items_id:ID(m)}
                """
        cls.__session__ = cls.createSession()
        return setInList(results=cls.__session__.run(query).values(), item="items_id")


    @classmethod
    def getAllUser(cls):
        query = """
                    MATCH (n:User)
                    return {user_id:ID(n)}
                """
        cls.__session__ = cls.createSession()
        return setInList(results=cls.__session__.run(query).values(), item="user_id")

    @classmethod
    def getUserRatingProduct(cls):
        query = """
                    MATCH (n:User)-[r:Notes]->(m:Produit) RETURN
                    {user_id:ID(n), items_id:ID(m), rating:r.rating}
                """
        cls.__session__ = cls.createSession()
        return setInDataFrame(cls.__session__.run(query))

    @classmethod
    def getAllProductWithName(cls):
        query = """
                    MATCH (n:Produit)
                    RETURN {id : ID(n), nom: n.designation}
                """
        cls.__session__ = cls.createSession()
        return setInDict(cls.__session__.run(query))
