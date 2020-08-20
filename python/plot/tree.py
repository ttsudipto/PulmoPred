from ..ml.pickler import load_saved_models
from ..config import config
from ..config import ml_config as mlc
from sklearn.tree import export_graphviz
from pathlib import Path
import numpy as np

path_prefix = config.get_OUTPUT_PATH() + 'graphs/'
n_graphs = 20

#inp = np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape(1,-1)

def get_version() :
    import sklearn
    #print(sklearn.__version__)
    return sklearn.__version__+'/'

class GraphGenerator :
    
    def __init__(self, model):
        self.RF_estimator = model.total_estimator
        self.trees = self.RF_estimator.estimators_
        self.graphviz_params = dict()
        self.init_graphviz_params()
        self.generate_graphs()
    
    def init_graphviz_params(self) :
        self.graphviz_params['feature_names'] = ['FEV1PreVal','FEV1PrePercent','FEV1PostVal','FEV1PostPercent','FVCPreVal','FVCPrePercent','FVCPostVal','FVCPostPercent','FEFPreVal','FEFPrePercent','FEFPostVal','FEFPostPercent']
        self.graphviz_params['node_ids'] = True
        self.graphviz_params['class_names'] = ['Non-obstructive', 'Obstructive']
    
    def generate_graphs(self) :
        self.graphs = []
        for tree in self.trees :
            self.graphs.append(export_graphviz(tree, **self.graphviz_params))
    
    def save_graphs(self, index) :
        prefix = path_prefix + get_version() + 'dot/' + str(index) + '/'
        Path(prefix).mkdir(parents=True, exist_ok=True)
        for i in range(len(self.graphs)) :
            f = open(prefix + str(i) + '.dot', 'w')
            f.write(self.graphs[i])
            f.close()

def save() :
    models = load_saved_models('RF')
    for i in range(len(models)) :
        gen = GraphGenerator(models[i])
        gen.save_graphs(i)
        #print(gen.graphs[4])

def load_graphs() :
    graphs = []
    for i in range(mlc.get_n_US_folds('PFT')) :
        g = []
        prefix = path_prefix + get_version() + 'dot/' + str(i) + '/'
        for j in range(n_graphs) :
            f = open(prefix + str(j) + '.dot', 'r')
            g.append(f.read())
            f.close()
        graphs.append(g)
    return graphs

def load_and_verify() :
    loaded_graphs = load_graphs()
    models = load_saved_models('RF')
    for i in range(len(models)) :
        gen = GraphGenerator(models[i])
        generated_graphs = gen.graphs
        for j in range(len(generated_graphs)) :
            if generated_graphs[j] != loaded_graphs[i][j] :
                print('Mismatch : Model -> ' + str(i) + ', graph -> ' + str(j))
                return False
    print('Ok')
    return True

save()
load_and_verify()
