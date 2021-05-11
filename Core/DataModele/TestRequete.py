from DatabaseAccess import Database

class Requete:

    connection = Database.getConnection("bolt://localhost:7687", "test_user", "root")
    __session__ = None

    def __init__(self):
        pass

    @classmethod
    def addSession(cls):
        return cls.connection.session()

    @classmethod
    def create_node(cls, age, nom):
        session = cls.addSession()
        return session.run("CREATE (n:Person {age:$age, nom:$nom}) RETURN n", age=age, nom=nom)

rq = Requete.create_node(22, "Testxx-123")
print(rq)
