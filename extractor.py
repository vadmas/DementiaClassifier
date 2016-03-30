import nltk
import subprocess
import threading
import requests
import re


class tree_node():

    def __init__(self, type):
        self.type = type
        self.children = []

    def addChild(self, node):
        self.children.append(node)


class StanfordServerThread(threading.Thread):

    def __init__(self, port = 9000):
        self.stdout = None
        self.stderr = None
        self.port = port
        self.p = None
        threading.Thread.__init__(self)

    def run(self):
        server_cmd = ['java', '-Xmx4g', '-cp', 'stanford/stanford-corenlp-full-2015-12-09/*',
                      'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', '-port', str(self.port)]
        self.p = subprocess.Popen(server_cmd)

    def stop_server(self):
        self.p.kill()



def start_stanford_server(port = 9000):
    stanfordServerThread = StanfordServerThread()
    stanfordServerThread.start()
    return stanfordServerThread


def get_parse_tree(sentences, port = 9000):
    #raw = sentence['raw']
    #pattern = '[a-zA-Z]*=\\s'
    #re.sub(pattern, '', raw)
    r = requests.post('http://localhost:' + str(port) + '/?properties={\"annotators\":\"parse\",\"outputFormat\":\"json\"}', data = sentence)
    json_obj = r.json()
    sentences = len(json_obj['sentences'])
    trees = []
    for sentence in sentences:
        trees = json_obj['sentences'][sentence]['parse']
    return trees


def build_tree(parse_tree):
    node_stack = []
    build_node = False
    node_type = ''
    root_node = None
    for ch in parse_tree:
        # If we encounter a ( character, start building a node
        if ch == '(':
            build_node = True
            continue
        if ch.isspace() and build_node:
            # Finished building node
            node = tree_node(node_type)
            node_stack.append(node)
            build_node = False
            node_type = ''
            continue
        if ch == ')':
            # pop from the stack and add it to the children for the node before it
            popped_node = node_stack.pop()
            if len(node_stack) > 0:
                parent = node_stack[-1]
                parent.addChild(popped_node)
            else:
                root_node = popped_node
            continue
        if build_node:
            node_type = node_type + ch
            continue
    return root_node


def get_height_of_tree(tree_node):
    depth = 0
    for children in tree_node.children:
        depth += get_height_of_tree(children)
    return depth


def get_count_of_parent_child(child_type, parent_type, tree_node, prev_type = None):
    curr_type = tree_node.type
    count = 0
    if prev_type == parent_type and curr_type == child_type:
        count = 1
    for children in tree_node.children:
        count += get_count_of_parent_child(child_type, parent_type, children, curr_type)
    return count


def get_NP_2_PRP(tree_node):
    return get_count_of_parent_child('PRP', 'NP', tree_node)


def get_ADVP_2_RB(tree_node):
    return get_count_of_parent_child('ADVP', 'RP', tree_node)


def get_NP_2_DTNN(tree_node):
    return get_count_of_parent_child('NP', 'DT_NN', tree_node)


def get_VP_2_AUXVP(tree_node):
    return get_count_of_parent_child('VP', 'AUX_VP', tree_node)


def get_VP_2_VBG(tree_node):
    return get_count_of_parent_child('VP', 'VBG', tree_node)


def get_VP_2_VBGPP(tree_node):
    return get_count_of_parent_child('VP', 'VBG_PP', tree_node)


def get_VP_2_AUXADJP(tree_node):
    return get_count_of_parent_child('VP', 'AUX_ADJP', tree_node)


def get_VP_2_AUX(tree_node):
    return get_count_of_parent_child('VP', 'AUX', tree_node)


def get_VP_2_VBDNP(tree_node):
    return get_count_of_parent_child('VP', 'VBD_NP', tree_node)


def get_INTJ_2_UH(tree_node):
    return get_count_of_parent_child('INTJ', 'UH', tree_node)


if __name__ == '__main__':
    thread = start_stanford_server() # Start the server
    tree = get_parse_tree('The quick brown fox jumped over the lazy dog.')
    build_tree(tree)
    thread.stop_server()
    #build_tree('u(ROOT\n  (S\n    (NP (DT The) (JJ quick) (JJ brown) (NN fox))\n    (VP (VBD jumped)\n      (PP (IN over)\n        (NP (DT the) (JJ lazy) (NN dog))))\n    (. .)))')
