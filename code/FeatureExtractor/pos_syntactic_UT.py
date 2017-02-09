import unittest
import pos_syntactic as ps

# Building test tree
root_node = ps.tree_node('ROOT')
ch_1 = ps.tree_node('NP')
ch_2 = ps.tree_node('VP')
ch_3 = ps.tree_node('ADJP')
ch_4 = ps.tree_node('ADVP')
ch_5 = ps.tree_node('ADVP')
ch_6 = ps.tree_node('NP')
ch_2_1 = ps.tree_node('PP')
ch_2_2 = ps.tree_node('VP')
ch_2_2_1 = ps.tree_node('VBD', 'phrase') # LEAF

ch_2_2.addChild(ch_2_2_1)
ch_2.addChild(ch_2_1)
ch_2.addChild(ch_2_2)
root_node.addChild(ch_1)
root_node.addChild(ch_2)
root_node.addChild(ch_3)
root_node.addChild(ch_4)
root_node.addChild(ch_5)
root_node.addChild(ch_6)

CFG_start = {
    "ADJP": 0,
    "ADVP": 0,
    "CONJP": 0,
    "FRAG": 0,
    "INTJ": 0,
    "LST": 0,
    "NAC": 0,
    "NP": 0,
    "NX": 0,
    "PP": 0,
    "PRN": 0,
    "PRT": 0,
    "QP": 0,
    "RRC": 0,
    "UCP": 0,
    "VP": 0,
    "WHADJP": 0,
    "WHAVP": 0,
    "WHNP": 0,
    "WHPP": 0,
    "X": 0
}


class Test_POS_Syntactic(unittest.TestCase):

    def test_tree_height(self):
        self.assertEqual(ps.get_height_of_tree(root_node), 4)


    def test_number_of_nodes(self):
        self.assertEqual(ps.get_number_of_nodes_in_tree(root_node), 10)

    def test_CFG(self):
        CFG_counts = {
            "ADJP": 1,
            "ADVP": 2,
            "CONJP": 0,
            "FRAG": 0,
            "INTJ": 0,
            "LST": 0,
            "NAC": 0,
            "NP": 2,
            "NX": 0,
            "PP": 1,
            "PRN": 0,
            "PRT": 0,
            "QP": 0,
            "RRC": 0,
            "UCP": 0,
            "VP": 2,
            "WHADJP": 0,
            "WHAVP": 0,
            "WHNP": 0,
            "WHPP": 0,
            "X": 0
        }
        self.assertEqual(CFG_counts, ps.get_CFG_counts(root_node, CFG_start))



if __name__ == '__main__':
    unittest.main()