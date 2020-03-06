import json
import sys
from input_wrapper import Input
#from ... import model
#import input_wrapper

def convert_to_float(s) :
    if s == '' :
        return 0.0
    else :
        return float(s)

def parse_json(j_string) :
    print(j_string)
    json_dict = json.loads(j_string)
    inp = Input()
    params = inp.get_all_params()
    for p in params :
        inp.add_value(convert_to_float(json_dict[p]))
    return inp

#print(sys.path)
inp = parse_json(sys.argv[1])
for i in range(inp.param_length) :
    print(inp.get_param(i) + ' = ' + str(inp.get_value(i)))
