from threading import Thread, Lock
from holoclean.dataengine import DataEngine

printLock = Lock()


class ModuleThreading:

    def __init__(self, table_name, session, worker_count=2):
        self.dataengine = DataEngine(session.holo_env)
        self.worker_count = worker_count
        self.table_name = table_name
        self.session = session
        self.query_list_lock = Lock()
        self.threads = []

        for i in range(self.worker_count):
            query_for_table = "CREATE TABLE " + self.table_name + \
                              "_" + str(i) + " (vid INT, assigned_val INT," \
                              " feature INT ,count INT);"
            self.dataengine.query(query_for_table)
        return

    def run_queries(self, queries):
        """
        run every query in a list in parallel
        :param queries: list of queries to be executed
        """

        # Create table for each thread and start running the thread
        for i in range(self.worker_count):
            thread = (QueryThread(
                queries, i, self.table_name, self.query_list_lock, self.session))
            self.threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for i in range(self.worker_count):
            self.threads[i].join()

        # Union threads tables together (Optional)

        return

    def retrieve(self):
        result = None
        name = self.table_name
        for i in range(self.worker_count):
            query = "SELECT * FROM " + name + "_" + str(i)
            if result is None:
                result = self.dataengine.query(query, 1)
            else:
                result = result.union(self.dataengine.query(query, 1))
        return result


class QueryThread(Thread):

    def __init__(self, queries, id, table_name, lock, session):
        Thread.__init__(self)
        self.lock = lock
        self.queries = queries
        self.dataengine = self.dataengine = DataEngine(session.holo_env)
        self.id = id
        self.table_name = table_name

    def run(self):
        """
        Retrieves queries from list and runs them until list is empty
        :return:
        """
        while True:
            self.lock.acquire()
            if len(self.queries) == 0:
                self.lock.release()
                break
            query = self.queries.pop()
            self.lock.release()
            insert_query = "INSERT INTO " + self.table_name + "_"+str(self.id)+"(" + query + ");"

            self.dataengine.query(insert_query)
