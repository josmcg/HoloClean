class Augmentor:
    def __init__(self,labeled_set, dataset, augmentation_rules):
        """
        Initialize an augmentation dataset handler
        :param labeled_set: a set of labeled training data
        :param augmentation_rules: a set of functions
            which take a an example [index, attribute, value, label] and a dataset (frame) and
            produce more cells with new labels
        :param dataset: a dataframe of the current dataset
        """

        self.augmentation_rules = augmentation_rules
        self.dataset = dataset
        # Todo this is essentially psuedocode
        self.labeled_set = pyspark.dataframe()
        for example in labeled_set:
            self.add_example(example)


    def add_example(self, example):
        """
        add an example to the labeled set, and apply augmentation rules to it
        :param example:
        :return:
        """
        self.labeled_set.append(example)
        for f in self.augmentation_rules:
            self.labeled_set.append(f(example, self.dataset))

    def batches(self):
        """
        Returns a dataloader interface to the labeled set for training
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """
        pass