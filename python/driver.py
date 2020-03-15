import json
import sys
from .input_wrapper import Input
from .ml.pickler import load_saved_models
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
    return inp

def classify(models, inp_vector) :
    output = dict()
    score_sum = 0
    threshold_sum = 0
    m_id = 0
    for m in models :
        threshold_sum = threshold_sum + m.optimal_threshold
        model_score_sum = 0
        for i in range(m.n_folds) :
            y_pred = m.get_decision_score(m.estimators[i], inp_vector)
            model_score_sum = model_score_sum + y_pred[0]
        score_sum = score_sum + (model_score_sum / float(m.n_folds))
        #print('Model score : ' + str((model_score_sum / float(m.n_folds))))
        #print('Model threshold : ' + str(m.optimal_threshold))
        output['score'+str(m_id)] = str((model_score_sum / float(m.n_folds)))
        output['threshold'+str(m_id)] = str(m.optimal_threshold)
        if (model_score_sum / float(m.n_folds)) > m.optimal_threshold :
            output['predicted_label'+str(m_id)] = 1
        else :
            output['predicted_label'+str(m_id)] = 0
        m_id = m_id + 1
    avg_threshold = threshold_sum / float(len(models))
    avg_score = score_sum / float(len(models))
    #print('Mean Score : ' + str(avg_score))
    #print('Mean Threshold : ' + str(avg_threshold))
    #if avg_score < avg_threshold :
        #print('Predicted class : 0')
    #else :
        #print('Predicted class : 1')
    print(json.dumps(output))

#print(sys.path)
inp = parse_json(sys.argv[1])
#for i in range(inp.param_length) :
    #print(inp.get_param(i) + ' = ' + str(inp.get_value(i)))

models = load_saved_models('SVM')
classify(models, inp.get_ndarray())
print(len(models))
