from neo4j import GraphDatabase
#A design pattern type singleton for access in database

class Database:
    #Static variable
    db_driver = None

    def __init__(self):
        if Database.db_driver is not None:
            raise Exception("Class not re-instanced")
        else:
            Database.db_driver = self

    @classmethod
    def getConnection(cls, uri=None, user=None, password=None):
        if cls.db_driver is None:
            cls.db_driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls.db_driver

# db = Database.getConnection("bolt://localhost:7687", "test_user", "root")
# db2 = Database.getConnection()
#print(db)
#print(db2)
