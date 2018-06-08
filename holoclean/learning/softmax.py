import torch
from torch.nn import Parameter, ParameterList
from torch.autograd import Variable
from torch import optim
from torch.nn.functional import softmax
from pyspark.sql.types import *
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F



class LogReg(torch.nn.Module):
    """
    Class to generate weights
    """

    def _setup_weights(self):
        """
        Initializes weight tensor with random values
        ties init and dc weights if specified
        :return: Null
        """
        torch.manual_seed(42)
        # setup init
        self.weight_tensors = ParameterList()
        self.tensor_tuple = ()
        self.feature_type = []
        self.W = None
        for featurizer in self.featurizers:
            self.feature_type.append(featurizer.type)
            if featurizer.type == 1:
                signals_W = Parameter(torch.randn(featurizer.M,
                                                  1).expand(-1,
                                                            self.output_dim))
            elif featurizer.type == 0:
                signals_W = Parameter(torch.randn(featurizer.M,self.output_dim))
            self.weight_tensors.append(signals_W)
        return

    def __init__(self, featurizers, output_dim,
                 tie_init, tie_dc):
        """
        Constructor for our logistic regression
        :param input_dim_non_dc: number of init + cooccur features
        :param input_dim_dc: number of dc features
        :param output_dim: number of classes
        :param tie_init: boolean, determines weight tying for init features
        :param tie_dc: boolean, determines weight tying for dc features
        """
        super(LogReg, self).__init__()

        self.featurizers = featurizers

        self.output_dim = output_dim

        self.tie_init = tie_init
        self.tie_dc = tie_dc

        self._setup_weights()

    def forward(self, X, index, mask):
        """
        Runs the forward pass of our logreg.
        :param X: values of the features
        :param index: indices to mask at
        :param mask: tensor to remove possibility of choosing unused class
        :return: output - X * W after masking
        """

        # Reties the weights - need to do on every pass

        self.concat_weights()

        # Calculates n x l matrix output
        output = X.mul(self.W)
        output = output.sum(1)
        # Changes values to extremely negative at specified indices
        if index is not None and mask is not None:
            output.index_add_(0, index, mask)
        return output

    def concat_weights(self):
        """
        Reties the weight
        """
        for feature_index in range(0, len(self.weight_tensors)):
            if self.feature_type[feature_index] == 1:
                tensor = self.weight_tensors[feature_index].expand(
                    -1, self.output_dim)
            else:
                tensor = self.weight_tensors[feature_index]
            if feature_index == 0:
                self.W = tensor + 0
            else:
                self.W = torch.cat((self.W, tensor), 0)



class SoftMax:

    def __init__(self, session):
        """
        Constructor for our softmax model
        :param X_training: x tensor used for training the model
        :param session: session object
        """
        self.session = session
        self.dataengine = session.holo_env.dataengine
        self.dataset = session.dataset
        self.holo_obj = session.holo_env
        self.spark_session = self.holo_obj.spark_session
        dataframe_offset = self .dataengine.get_table_to_dataframe(
            "Dimensions_clean", self.dataset)
        list = dataframe_offset.collect()
        dimension_dict = {}
        for dimension in list:
            dimension_dict[dimension['dimension']] = dimension['length']

        # X Tensor Dimensions (N * M * L)
        self.N = dimension_dict['N']
        self.L = dimension_dict['L']

        self.testM = None
        self.testN = None
        self.testL = None

        # pytorch tensors
        self.mask = None
        self.testmask = None
        self.setupMask()
        self.Y = None
        self.grdt = None
        self._setupY()
        self.model = None

    # Will create the Y tensor of size NxL
    def _setupY(self):
        """
        Initializes a y tensor to compare to our model's output
        :return: Null
        """
        possible_values = self.dataengine .get_table_to_dataframe(
            "Observed_Possible_values_clean", self.dataset) .collect()
        self.Y = torch.zeros(self.N, 1).type(torch.LongTensor)
        for value in possible_values:
            self.Y[value.vid - 1, 0] = value.domain_id - 1
        self.grdt = self.Y.numpy().flatten()
        return

    def setupMask(self, clean=1, N=1, L=1):
        """
        Initializes a masking tensor for ignoring impossible classes
        :param clean: 1 if clean cells, 0 if don't-know
        :param N: number of examples
        :param L: number of classes
        :return: masking tensor
        """
        lookup = "Kij_lookup_clean" if clean else "Kij_lookup_dk"
        N = self.N if clean else N
        L = self.L if clean else L
        K_ij_lookup = self.dataengine.get_table_to_dataframe(
            lookup, self.dataset).select("vid", "k_ij").collect()
        mask = torch.zeros(N, L)
        for domain in K_ij_lookup:
            if domain.k_ij < L:
                mask[domain.vid - 1, domain.k_ij:] = -10e6
        if clean:
            self.mask = mask
        else:
            self.testmask = mask
        return mask

    def build_model(self,  featurizers,
                    output_dim, tie_init=True, tie_DC=True):
        """
        Initializes the logreg part of our model
        :param featurizers: list of featurizers
        :param input_dim_dc: number of dc features
        :param output_dim: number of classes
        :param tie_init: boolean to decide weight tying for init features
        :param tie_DC: boolean to decide weight tying for dc features
        :return: newly created LogReg model
        """
        model = LogReg(
            featurizers,
            output_dim,
            tie_init,
            tie_DC)
        return model

    def train(self, model, loss, optimizer, x_val, y_val, mask=None):
        """
        Trains our model on the clean cells
        :param model: logistic regression model
        :param loss: loss function used for evaluating performance
        :param optimizer: optimizer for our neural net
        :param x_val: x tensor - features
        :param y_val: y tensor - output for comparison
        :param mask: masking tensor
        :return: cost of traininng
        """
        x = Variable(x_val, requires_grad=False)
        y = Variable(y_val, requires_grad=False)

        if mask is not None:
            mask = Variable(mask, requires_grad=False)

        index = torch.LongTensor(range(x_val.size()[0]))
        index = Variable(index, requires_grad=False)

        # Reset gradient
        optimizer.zero_grad()

        # Forward
        fx = model.forward(x, index, mask)

        output = loss.forward(fx, y.squeeze(1))

        # Backward
        output.backward()

        # Update parameters
        optimizer.step()

        return output.data[0]

    def predict(self, model, x_val, mask=None):
        """
        Runs our model on the test set
        :param model: trained logreg model
        :param x_val: test x tensor
        :param mask: masking tensor to restrict domain
        :return: predicted classes with probabilities
        """
        x = Variable(x_val, requires_grad=False)

        index = torch.LongTensor(range(x_val.size()[0]))
        index = Variable(index, requires_grad=False)

        if mask is not None:
            mask = Variable(mask, requires_grad=False)

        output = model.forward(x, index, mask)
        output = softmax(output, 1)

        return output

    def prediction(self, featurizers, model,N,L, mask=None ):
        """
        Runs our model on the test set
        :param model: trained logreg model
        :param x_val: test x tensor
        :param mask: masking tensor to restrict domain
        :return: predicted classes with probabilities
        """
        x_val = None
        for featurizer in featurizers:
            sub_tensor = featurizer.create_tensor(0, N, L)
            if x_val is None:
                x_val = sub_tensor
            else:
                x_val = torch.cat((x_val,sub_tensor),1)

        x_val = F.normalize(x_val, p=2, dim=1)

        x = Variable(x_val, requires_grad=False)

        index = torch.LongTensor(range(x_val.size()[0]))
        index = Variable(index, requires_grad=False)

        if mask is not None:
            mask = Variable(mask, requires_grad=False)

        output = model.forward(x, index, mask)
        output = softmax(output, 1)

        return output

    def logreg(self, featurizers):
        """
        Trains our model on clean cells and predicts vals for clean cells
        :return: predictions
        """
        # n_examples, n_features, n_classes = self.X.size()
        self.model = self.build_model(
            featurizers, self.L)
        loss = torch.nn.CrossEntropyLoss(size_average=True)
        need_update = False
        list_parameters = None
        for featurizer in featurizers:
            if featurizer.update_flag:
                need_update = True
            if list_parameters is None:
                list_parameters = list(featurizer.parameters())
            else:
                list_parameters = list_parameters + list(featurizer.parameters())
        optimizer = optim.SGD(
            list(self.model.parameters()) + list_parameters,
            lr=self.holo_obj.learning_rate,
            momentum=self.holo_obj.momentum,
            weight_decay=self.holo_obj.weight_decay)

        # Experiment with different batch sizes. no hard rule on this

        # create x tensor
        X = self.create_tensor(featurizers)
        X = F.normalize(X, p=2, dim=1)
        batch_size = self.holo_obj.batch_size
        for i in tqdm(range(self.holo_obj.learning_iterations)):
            cost = 0.
            num_batches = self.N // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size

                cost += self.train(self.model,
                                   loss,
                                   optimizer,
                                   X[start:end],
                                   self.Y[start:end],
                                   self.mask[start:end])
                if need_update:
                    X = self.create_tensor(featurizers)
                    X = F.normalize(X, p=2, dim=1)

            predY = self.predict(self.model, X, self.mask)
            map = predY.data.numpy().argmax(axis=1)

            if self.holo_obj.verbose:
                print("Epoch %d, cost = %f, acc = %.2f%%" %
                      (i + 1, cost / num_batches,
                       100. * np.mean(map == self.grdt)))
        return self.predict(self.model, X, self.mask)

    def create_tensor(self,featurizers):
        """
        This method creates the X tensor that we will use in our model

        :param featurizers: a list of all the pytorch modules that we use as
        featurizers
\       :return: X tensors
        """
        X = None
        for featurizer in featurizers:
            sub_tensor = featurizer.forward()
            if X is None:
                X = sub_tensor
            else:
                X = torch.cat((X,sub_tensor),1)
        return X

    def save_prediction(self, Y):
        """
        Stores our predicted values in the database
        :param Y: tensor with probability for each class
        :return: Null
        """
        k_inferred = self.session.holo_env.k_inferred

        if k_inferred > Y.size()[1]:
            k_inferred = Y.size()[1]

        max_result = torch.topk(Y,k_inferred,1)
        max_indexes = max_result[1].data.tolist()
        max_prob = max_result[0].data.tolist()

        vid_to_value = []
        df_possible_values = self.dataengine.get_table_to_dataframe(
            'Possible_values_dk', self.dataset).select(
            "vid", "attr_name", "attr_val", "tid", "domain_id")

        # Save predictions upt to the specified k unless Prob = 0.0
        for i in range(len(max_indexes)):
                for j in range(k_inferred):
                    if max_prob[i][j]:
                        vid_to_value.append([i + 1, max_indexes[i][j] + 1,
                                             max_prob[i][j]])
        df_vid_to_value = self.spark_session.createDataFrame(
            vid_to_value, StructType([
                StructField("vid1", IntegerType(), False),
                StructField("domain_id1", IntegerType(), False),
                StructField("probability", DoubleType(), False)
            ])
        )
        df1 = df_vid_to_value
        df2 = df_possible_values
        df_inference = df1.join(
            df2, [
                df1.vid1 == df2.vid,
                df1.domain_id1 == df2.domain_id], 'inner')\
            .drop("vid1","domain_id1")

        self.session.inferred_values = df_inference
        self.session.holo_env.logger.info\
            ("The Inferred_values Data frame has been created")
        self.session.holo_env.logger.info("  ")
        return

    def log_weights(self):
        """
        Writes weights in the logger
        :return: Null
        """
        self.model.concat_weights()
        weights = self.model.W
        self.session.holo_env.logger.info("Tensor weights")
        count = 0
        for weight in \
                torch.index_select(
                    weights, 1, Variable(torch.LongTensor([0]))
                ):

            count += 1
            msg = "Feature " + str(count) + ": " + str(weight[0].data[0])
            self.session.holo_env.logger.info(msg)

        return
