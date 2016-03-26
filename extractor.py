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


def get_parse_tree(sentence, port = 9000):
    #raw = sentence['raw']
    #pattern = '[a-zA-Z]*=\\s'
    #re.sub(pattern, '', raw)
    r = requests.post('http://localhost:' + str(port) + '/?properties={\"annotators\":\"parse\",\"outputFormat\":\"json\"}', data = sentence)
    json_obj = r.json()
    tree = json_obj['sentences'][0]['parse']
    return tree


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


if __name__ == '__main__':
    thread = start_stanford_server() # Start the server
    tree = get_parse_tree('The quick brown fox jumped over the lazy dog.')
    build_tree(tree)
    thread.stop_server()
    #build_tree('u(ROOT\n  (S\n    (NP (DT The) (JJ quick) (JJ brown) (NN fox))\n    (VP (VBD jumped)\n      (PP (IN over)\n        (NP (DT the) (JJ lazy) (NN dog))))\n    (. .)))')
