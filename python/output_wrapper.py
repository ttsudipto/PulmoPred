from json import JSONEncoder


class Output :
    def __init__(self, classifier):
        self.classifier = classifier
        self.version = None
        self.labels = []
        self.scores = []
        self.probas = []
        
        self.paths = []
        
        self.thresholds = []
        self.positivenesses = []
        self.negativenesses = []
        
        self.coeffs = []
        self.biases = []


class OutputEncoder(JSONEncoder) :
    def default(self, obj):
        if isinstance(obj, Output) :
            out_dict = dict()
            out_dict['classifier'] = obj.classifier
            out_dict['version'] = obj.version
            out_dict['labels'] = obj.labels
            out_dict['scores'] = obj.scores
            out_dict['probas'] = obj.probas
            out_dict['paths'] = obj.paths
            out_dict['thresholds'] = obj.thresholds
            out_dict['positivenesses'] = obj.positivenesses
            out_dict['negativenesses'] = obj.negativenesses
            out_dict['coeffs'] = obj.coeffs
            out_dict['biases'] = obj.biases
            return out_dict
        return JSONEncoder.default(self, obj)
