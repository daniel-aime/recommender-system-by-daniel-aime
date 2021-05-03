from neo4j import GraphDatabase

class ConnectDB:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user,password))


    def close_connection(self):
        self.driver.close()


    def _exec_query(self, name, pays_origine):
        with self.driver.session() as session:
            greeting = session.write_transaction(self._create_new_node, name, pays_origine)
            print(greeting)

    @staticmethod
    def _create_new_node(tx, name, pays_origine):
        result = tx.run("create (a:Person "
               "{name:$name, pays_origine:$pays_origine})"
               " return a", name=name, pays_origine=pays_origine)
        return result.single()[0]

if __name__ == '__main__':
    testsr = ConnectDB("bolt://localhost:7687", "test_user", "root")
    testsr._exec_query("Nancy", "Madagascar")
    testsr.close_connection()
