from utils.common_variables import *
from utils.checkpoint import *
import networkx as nx
import uunet.multinet as ml
import pandas as pd
from utils.log_manager.log_manager import *

file_name = os.path.splitext(os.path.basename(__file__))[0]

class ConversionManager:
    def __init__(self):
        self.lm = LogManager('main')
        self.ch = Checkpoint()

    def to_df(self, d):
        return pd.DataFrame.from_dict(d)

    def from_df_to_edge_list(self, df):
        # convert to a list of tuples. to_records convert to a list of numpy array instead
        return list(df.itertuples(index=False, name=None))

    def from_edge_list_to_df(self, edge_list, path):
        # dict_edge_list = {"userId1": [], "userId2": [], "weight": []}
        # for e in filtered_edge_list:
        #     dict_edge_list["userId1"].append(e[tuple_index['userId1']])
        #     dict_edge_list["userId2"].append(e[tuple_index['userId2']])
        #     dict_edge_list["weight"].append(e[tuple_index['weight']])
        # df = pd.DataFrame(dict_edge_list)

        # Convert to DataFrame with specified column names
        columns = list(tuple_index.keys())
        # columns = columns[0:4] # alpha of backbone not used
        df = pd.DataFrame(edge_list, columns=columns)
        self.ch.save_dataframe(df, path)

    def from_graph_to_gephi(self, G, path):
        # path = self.__get_path(filename, dir_path, add_prefix)
        nx.write_gexf(G, path)
        self.lm.printl(f"{file_name}: graph_to_gephi: {path}")

    def from_edge_list_to_graph(self, edge_list):
        # Create an empty graph
        G = nx.Graph()
        # Add edges with attributes
        for edge in edge_list:
            G.add_edge(edge[tuple_index[NODE1_VAR]], edge[tuple_index[NODE2_VAR]], w_=edge[tuple_index[W_VAR]],
                       nActions=edge[tuple_index[NA_VAR]], twCount=edge[tuple_index[TW_VAR]])

        return G

    def add_layer_multiplex_network(self, MG, G, layer):
        # TODO: Add node attributes, add parametric attribute type
        ml.add_nx_layer(MG, G, layer)
        edges_dict = {'from_actor': [], 'from_layer': [], 'to_actor': [], 'to_layer': []}
        attr_list = list(tuple_index.keys())[2:]

        values_attr_dict = {}
        for attr in attr_list:
            values_attr_dict[attr] = []

        # add edge attributes (type numeric) in the multiplex network to the layer "layer"
        ml.add_attributes(MG, attr_list, target="edge", type='numeric', layer=layer)

        # convert the graph to the format
        # {'from_actor': ['1000044012202594306'], 'from_layer': ['co-retweet'], 'to_actor': ['1007223608320647168'], 'to_layer': ['co-retweet']}
        for edge in G.edges(data=True):
            edges_dict['from_actor'].append(edge[0])
            edges_dict['from_layer'].append(layer)
            edges_dict['to_actor'].append(edge[1])
            edges_dict['to_layer'].append(layer)

            attributes_dict = edge[2]
            for attr in attr_list:
                values_attr_dict[attr].append(attributes_dict[attr])

        # first two elements of tuple_index are userId1, userId2
        for attr in attr_list:
            ml.set_values(MG, attr, edges=edges_dict, values=values_attr_dict[attr])
        self.lm.printl(f"{layer} added to the multiplex network.")
        return MG