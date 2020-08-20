from ..config import config
from ..config import ml_config as mlc
from graphviz import Source, pipe
from pathlib import Path

n_graphs = 20
path_prefix = config.get_OUTPUT_PATH() + 'graphs/'

def get_version() :
    import sklearn
    #print(sklearn.__version__)
    return sklearn.__version__+'/'

def read_DOT(filename) :
    f = open(filename, 'rb')
    graph = f.read()
    f.close()
    return graph

def write_SVG(filename, directory, data) :
    Path(directory).mkdir(parents=True, exist_ok=True)
    f = open(directory + filename, 'w')
    f.write(data)
    f.close()

def load_SVG(filename) :
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return data

def convert_dot_to_svg(version=None) :
    if version is None :
        version = get_version()
    #print(version)
    for i in range(mlc.get_n_US_folds('PFT')) :
        for j in range(n_graphs) :
            input_filename = path_prefix + version + 'dot/' + str(i) + '/' + str(j) + '.dot'
            op_format, engine = 'svg', 'dot'
            svg = pipe(engine, op_format, read_DOT(input_filename))
            #print(svg.decode('utf-8'))
            output_directory = path_prefix + version + 'svg/' + str(i) + '/'
            write_SVG(str(j) + '.svg', output_directory, svg.decode('utf-8'))

def load_and_verify(version=None) :
    if version is None :
        version = get_version()
    for i in range(mlc.get_n_US_folds('PFT')) :
        for j in range(n_graphs) :
            dot_filename = path_prefix + version + 'dot/' + str(i) + '/' + str(j) + '.dot'
            computed_svg = pipe('dot', 'svg', read_DOT(dot_filename))
            svg_filename = path_prefix + version + 'svg/' + str(i) + '/' + str(j) + '.svg'
            loaded_svg = load_SVG(svg_filename)
            if loaded_svg != computed_svg.decode('utf-8') :
                print('Not matching : Model - ' + str(i) + ', Graph - ' + str(j))
                return False
    print('Matched')
    return True

#convert_dot_to_svg()
load_and_verify()
