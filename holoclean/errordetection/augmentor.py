from holoclean.global_variables import GlobalVariables
from pyspark.sql.functions import col

class Augmentor:
    def __init__(self,ground_truth_path, session, augmentation_rules=None):
        """
        Initialize an augmentation dataset handler
        :param ground_truth: a subset of cleaned data, flattened
        :param augmentation_rules: a set of functions
            which take a an example [index, attribute, value, label] and a dataset (frame) and
            produce more cells with new labels
        :param session: a holoclean session object
        """

        self.augmentation_rules = augmentation_rules
        self.session = session
        self.ground_truth_path = ground_truth_path
        # Todo this is essentially psuedocode
        self.labeled_set = self._labeled_table()

    def add_example(self, example):
        """
        add an example to the labeled set, and apply augmentation rules to it
        :param example:
        :return:
        """
        self.labeled_set.append(example)
        for f in self.augmentation_rules:
            self.labeled_set.append(f(example, self.dataset))

    def _labeled_table(self):
        """
        flattens the ground truth into a labeled set
        :return:
        """
        flattened_init = self._flatten_init(self.session)
        ground_truth = self.read_ground_truth(self.session)
        joined = ground_truth\
            .withColumnRenamed("attr_val", "attr_val_clean")\
            .withColumnRenamed("rv_index", "c_rv_index")\
            .withColumnRenamed("rv_attr", "c_rv_attr")\
            .join(flattened_init.alias('init'), [col("c_rv_index") == col("init.rv_index")
            , col("c_rv_attr") == col("init.rv_attr")])
        joined = joined.drop("c_rv_attr", "c_rv_index", "attr_val_clean")
        labels = joined.withColumn("error",  ~ (col("attr_val") == col("attr_val_clean")))

        return labels

    def get(self):
        return self.labeled_set

    def _flatten_init(self, session):
        """
            given a holoclean session,generate a flattened init table
            """
        cols = session.dataset.get_schema("Init")
        cols.remove(GlobalVariables.index_name)
        # decide is we need to create a new table
        exists = False
        try:
            query_check = "SELECT * FROM " + session.dataset.table_specific_name("Init_flat")
            session.holo_env.dataengine.query(query_check)
            exists = True
        except:
            pass

        if not exists:
            query_create = "CREATE TABLE \
                            " + session.dataset.table_specific_name('Init_flat') \
                           + "( rv_index TEXT, \
                            rv_attr TEXT, attr_val TEXT);"
            session.holo_env.dataengine.query(query_create)
            # now with the table certainly created and empty, we populate it
            for col in cols:
                # get all cols
                query_get = "(SELECT DISTINCT t1." + GlobalVariables.index_name + \
                            " as rv_index " + \
                            ", \'" + col + "\' as rv_attr , t1." + \
                            col + " as attr_val" + \
                            "  FROM " + session.dataset.table_specific_name("Init") + \
                            " as t1)"
                query_insert = "INSERT INTO " + session.dataset.table_specific_name("Init_flat") + \
                               " " + query_get
                session.holo_env.dataengine.query(query_insert)
        else:
            pass
        return session.holo_env.dataengine.get_table_to_dataframe(
            'Init_flat', session.dataset)

    def read_ground_truth(self, session):
        """
        turns the path given at initialization into a dataframe
        :return: a pyspark dataframe
        """

        query_create = "CREATE TABLE " + session.dataset.table_specific_name("ground_truth") + \
                       "(rv_index TEXT, rv_attr TEXT, attr_val TEXT)"
        session.holo_env.dataengine.query(query_create)
        # parse the lines in the file
        with open(self.ground_truth_path) as handle:
            # use map when you implement this robustly
            for line in [line.strip() for line in handle][1:]:
                items = line.split(",")
                query_insert = "INSERT INTO {} VALUES ('{}', '{}', '{}')".format(
                    session.dataset.table_specific_name("ground_truth"),
                    items[0],
                    items[1],
                    items[2]
                )
                session.holo_env.dataengine.query(query_insert)

        return session.holo_env.dataengine.get_table_to_dataframe(
            'ground_truth', session.dataset)