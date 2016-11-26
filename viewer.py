import pydot
from PIL import Image


def walk_dictionaryv2(graph, dictionary, parent_node=None):
    '''
    Recursive plotting function for the decision tree stored as a dictionary
    '''

    for k in dictionary.keys():

        if parent_node is not None:

            from_name = parent_node.get_name().replace("\"", "") + '_' + str(k)
            from_label = str(k)

            node_from = pydot.Node(from_name, label=from_label)

            graph.add_edge(pydot.Edge(parent_node, node_from))

            if isinstance(dictionary[k], dict):  # if interim node

                walk_dictionaryv2(graph, dictionary[k], node_from)

            else:  # if leaf node
                to_name = str(k) + '_' + str(dictionary[k])  # unique name
                to_label = str(dictionary[k])

                node_to = pydot.Node(to_name, label=to_label, shape='box')
                graph.add_edge(pydot.Edge(node_from, node_to))

                # node_from.set_name(to_name)

        else:

            from_name = str(k)
            from_label = str(k)

            node_from = pydot.Node(from_name, label=from_label)
            walk_dictionaryv2(graph, dictionary[k], node_from)


def plot_tree(tree, name):

    # create pydot graph
    graph = pydot.Dot(graph_type='graph')
    walk_dictionaryv2(graph, tree)

    # write into png-file
    graph.write_png(name + ".png")

    # open image file
    img = Image.open(name + ".png")
    img.show()


def main():

    tree = {'salary': {'41k-45k': 'junior',
                       '46k-50k': {'department': {'marketing': 'senior',
                                                  'sales': 'senior',
                                                  'systems': 'junior'}},
                       '36k-40k': 'senior',
                       '26k-30k': 'junior',
                       '31k-35k': 'junior',
                       '66k-70k': 'senior'}}

    plot_tree(tree, "tree-img")


if __name__ == "__main__":
    main()
