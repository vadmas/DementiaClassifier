import unittest
import pos_syntactic as ps

# Building test tree
root_node = ps.tree_node('Blah')
ch_1 = ps.tree_node('Blah')
ch_2 = ps.tree_node('Blah')
ch_2_1 = ps.tree_node('Blah')
ch_2_2 = ps.tree_node('Blah')
ch_2_2_1 = ps.tree_node('Blah')

ch_2_2.addChild(ch_2_2_1)
ch_2.addChild(ch_2_1)
ch_2.addChild(ch_2_2)
root_node.addChild(ch_1)
root_node.addChild(ch_2)


class Test_POS_Syntactic(unittest.TestCase):

    def test_tree_height(self):
        self.assertEqual(ps.get_height_of_tree(root_node), 4)


    def test_number_of_nodes(self):
        self.assertEqual(ps.get_number_of_nodes_in_tree(root_node), 6)

if __name__ == '__main__':
    unittest.main()