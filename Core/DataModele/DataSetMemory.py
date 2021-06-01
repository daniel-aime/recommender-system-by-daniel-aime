from .DatabaseAccess import Database
from .DataFeaturing import *

class DataSetMemoryMethod:

    connection = Database.getConnection(uri="bolt://localhost:7687", user="test_user", password="root")
    __session__ = None
    __instance__ = None # noqa

    @staticmethod
    def getSingletonInstance():
        if DataSetMemoryMethod.__instance__ is None:
            DataSetMemoryMethod.__instance__ = DataSetMemoryMethod()
        return DataSetMemoryMethod.__instance__

    @classmethod
    def createSession(cls):
        return cls.connection.session()


    @classmethod
    def getProductUserRating(cls, user_id:int=None):
        query = """
                    MATCH (m:Movie) WHERE (:Person <id>=1)-[r:REVIEWED]->(m)
                """
