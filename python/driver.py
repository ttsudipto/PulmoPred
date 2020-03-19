import json
import sys
from statistics import mode
from .input_wrapper import Input
from .ml.pickler import load_saved_models
from .ml.density import compute_positiveness, compute_negativeness
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
    inp.set_estimator_id(json_dict['model_id'])
    return inp

def classify_with_SVM(inp_vector) :
    models = load_saved_models('SVM', with_CV=False)
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
        score = (model_score_sum / float(m.n_folds))
        output['score'+str(m_id)] = str(score)
        output['threshold'+str(m_id)] = str(m.optimal_threshold)
        output['positiveness'+str(m_id)] = str(compute_positiveness(m_id, score)*100)
        output['negativeness'+str(m_id)] = str(compute_negativeness(m_id, score)*100)
        if score > m.optimal_threshold :
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

def classify_with_RF(inp_vector) :
    models = load_saved_models('RF', with_CV=False)
    output = dict()
    m_id = 0
    for m in models :
        model_proba_sum = [0, 0]
        prediction_labels = []
        for i in range(m.n_folds) :
            y_pred = m.get_decision_score(m.estimators[i], inp_vector)
            model_proba_sum = [model_proba_sum[0] + y_pred[0][0], model_proba_sum[1] + y_pred[0][1]]
            prediction_labels.append(int(m.predict(m.estimators[i], inp_vector)[0]))
            #print(model_proba_sum)
        proba = [model_proba_sum[0] / float(m.n_folds), model_proba_sum[1] / float(m.n_folds)]
        #print(proba)
        output['proba'+str(m_id)] = str(max(proba))
        output['predicted_label'+str(m_id)] = mode(prediction_labels)
        m_id = m_id + 1
    print(json.dumps(output))

def classify_with_GNB(inp_vector) :
    classify_with_SVM(inp_vector)

inp = parse_json(sys.argv[1])
#for i in range(inp.param_length) :
    #print(inp.get_param(i) + ' = ' + str(inp.get_value(i)))

if inp.get_estimator_id() == 'SVM' :
    classify_with_SVM(inp.get_ndarray())
elif inp.get_estimator_id() == 'RF' :
    classify_with_RF(inp.get_ndarray())
elif inp.get_estimator_id() == 'GNB' :
    classify_with_GNB(inp.get_ndarray())
else :
    raise ValueError('Invalid classification model')
