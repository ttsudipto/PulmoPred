import json
import sys
from statistics import mode
from .input_wrapper import Input
from .output_wrapper import Output, OutputEncoder
from .ml.pickler import load_saved_models, get_version
from .ml.density import compute_positiveness, compute_negativeness
from .config import ml_config as mlc
#import input_wrapper

def convert_to_float(s) :
    if s == '' :
        return 0.0
    else :
        return float(s)

def parse_json(j_string) :
    #print(j_string)
    json_dict = json.loads(j_string)
    inp = Input()
    params = inp.get_all_params()
    for p in params :
        inp.add_value(convert_to_float(json_dict[p]))
    #inp.set_estimator_id(json_dict['model_id'])
    return inp

def get_decision_tree_paths(model, input_vector) :
    if mlc.is_RandomForest_id(model.estimator_id) :
        trees = model.total_estimator.estimators_
        paths = []
        for t in trees :
            paths.append(t.decision_path(input_vector).indices.tolist())
            #print(t.decision_path(input_vector).indices)
            #print()
        return paths
    else :
        return []

def classify_with_SVM(inp_vector) :
    models = load_saved_models('SVM')
    m_id = 0
    #score_sum = 0
    #threshold_sum = 0
    output = Output('SVM')
    for m in models :
        score = m.get_decision_score(m.total_estimator, inp_vector)[0]
        predicted_label = int(m.predict(m.total_estimator, inp_vector, m.optimal_threshold)[0])
        #score_sum = score_sum + score
        #threshold_sum = threshold_sum + m.optimal_threshold
        output.scores.append(str(score))
        output.thresholds.append(str(m.optimal_threshold))
        output.positivenesses.append(str(compute_positiveness(m_id, score)*100))
        output.negativenesses.append(str(compute_negativeness(m_id, score)*100))
        output.labels.append(str(predicted_label))
        m_id = m_id + 1
    #avg_score = score_sum / float(len(models))
    #avg_threshold = threshold_sum / float(len(models))
    print(json.dumps(output, cls=OutputEncoder))

def classify_with_RF(inp_vector) :
    models = load_saved_models('RF')
    output = Output('RF')
    output.version = get_version()
    m_id = 0
    for m in models :
        probas = m.get_decision_score(m.total_estimator, inp_vector)
        predicted_label = int(m.predict(m.total_estimator, inp_vector)[0])
        output.probas.append(str(probas[0][predicted_label]))
        output.labels.append(str(predicted_label))
        output.paths.append(get_decision_tree_paths(m, inp_vector))
        m_id = m_id + 1
    print(json.dumps(output, cls=OutputEncoder))

def classify_with_GNB(inp_vector) :
    models = load_saved_models('GNB')
    output = Output('GNB')
    m_id = 0
    for m in models :
        probas = m.get_decision_score(m.total_estimator, inp_vector)
        predicted_label = int(m.predict(m.total_estimator, inp_vector)[0])
        output.probas.append(str(probas[0][predicted_label]))
        output.labels.append(str(predicted_label))
        m_id = m_id + 1
    print(json.dumps(output, cls=OutputEncoder))

def classify_with_MLP(inp_vector) :
    models = load_saved_models('MLP')
    output = Output('MLP')
    m_id = 0
    for m in models :
        probas = m.get_decision_score(m.total_estimator, inp_vector)
        predicted_label = int(m.predict(m.total_estimator, inp_vector)[0])
        output.probas.append(str(probas[0][predicted_label]))
        output.labels.append(str(predicted_label))
        m_id = m_id + 1
    print(json.dumps(output, cls=OutputEncoder))


inp = parse_json(sys.argv[1])
#for i in range(inp.param_length) :
    #print(inp.get_param(i) + ' = ' + str(inp.get_value(i)))

if inp.get_estimator_id() == 'SVM' :
    classify_with_SVM(inp.get_ndarray())
elif inp.get_estimator_id() == 'RF' :
    classify_with_RF(inp.get_ndarray())
elif inp.get_estimator_id() == 'GNB' :
    classify_with_GNB(inp.get_ndarray())
elif inp.get_estimator_id() == 'MLP' :
    classify_with_MLP(inp.get_ndarray())
else :
    raise ValueError('Invalid classification model')
