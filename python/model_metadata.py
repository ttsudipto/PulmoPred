from model import Model
import copy

class ModelMetadata :
    def __init__(self, model, threshold) :
        self.train_indices = model.train_indices
        self.test_indices = model.test_indices
        self.estimator_id = model.estimator_id
        self.data = model.data
        self.target = model.target
        self.n_folds = model.n_folds
        self.SVM_params = model.SVM_params
        self.RF_params = model.RF_params
        self.GNB_params = model.GNB_params
        self.threshold = threshold

    def get_model(self) :
        model = Model(copy.deepcopy(self.estimator_id), self.data.copy(), self.target.copy(), k=self.n_folds, do_shuffle=False)
        model.train_indices = copy.deepcopy(self.train_indices)
        model.test_indices = copy.deepcopy(self.test_indices)
        model.SVM_params = copy.deepcopy(self.SVM_params)
        model.RF_params = copy.deepcopy(self.RF_params)
        model.GNB_params = copy.deepcopy(self.GNB_params)
        model.optimal_threshold = self.threshold
        return model
