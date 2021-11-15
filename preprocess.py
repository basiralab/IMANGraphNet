import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import torch
from config import N_TARGET_NODES_F, N_SOURCE_NODES_F,N_TARGET_NODES,N_SOURCE_NODES


def convert_vector_to_graph_RH(data):
    """
        convert subject vector to adjacency matrix then use it to create a graph
        edge_index:
        edge_attr:
        x:
    """

    data.reshape(1, N_SOURCE_NODES_F)
    # create adjacency matrix
    tri = np.zeros((N_SOURCE_NODES, N_SOURCE_NODES))
    tri[np.triu_indices(N_SOURCE_NODES, 1)] = data
    tri = tri + tri.T
    tri[np.diag_indices(N_SOURCE_NODES)] = 1

    edge_attr = torch.Tensor(tri).view(N_SOURCE_NODES**2, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    N_ROI = N_SOURCE_NODES

    pos_edge_index = torch.zeros(2, N_ROI * N_ROI)
    neg_edge_indexe = []
    # pos_edge_indexe = []
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

        # xx = torch.ones(160, 160, dtype=torch.float)

        x = torch.tensor(tri, dtype=torch.float)
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)

        return Data(x=x, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
def convert_vector_to_graph_HHR(data):
    """
        convert subject vector to adjacency matrix then use it to create a graph
        edge_index:
        edge_attr:
        x:
    """

    data.reshape(1, 35778)
    # create adjacency matrix
    tri = np.zeros((268, 268))
    tri[np.triu_indices(268, 1)] = data
    tri = tri + tri.T
    tri[np.diag_indices(268)] = 1

    edge_attr = torch.Tensor(tri).view(71824, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    N_ROI = 268

    pos_edge_index = torch.zeros(2, N_ROI * N_ROI)
    neg_edge_indexe = []
    # pos_edge_indexe = []
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

        # xx = torch.ones(268, 268, dtype=torch.float)

        x = torch.tensor(tri, dtype=torch.float)
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)

        return Data(x=x, pos_edge_index=pos_edge_index, edge_attr=edge_attr)

def convert_vector_to_graph_FC(data):
    """
        convert subject vector to adjacency matrix then use it to create a graph
        edge_index:
        edge_attr:
        x:
    """

    data.reshape(1, N_TARGET_NODES_F)
    # create adjacency matrix
    tri = np.zeros((N_TARGET_NODES, N_TARGET_NODES))
    tri[np.triu_indices(N_TARGET_NODES, 1)] = data
    tri = tri + tri.T
    tri[np.diag_indices(N_TARGET_NODES)] = 1

    edge_attr = torch.Tensor(tri).view(N_TARGET_NODES**2, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    counter = 0
    pos_counter = 0
    neg_counter = 0
    N_ROI = N_TARGET_NODES

    pos_edge_index = torch.zeros(2, N_ROI * N_ROI)
    neg_edge_indexe = []
    # pos_edge_indexe = []
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

        # xx = torch.ones(160, 160, dtype=torch.float)

        x = torch.tensor(tri, dtype=torch.float)
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)


    return Data(x=x, pos_edge_index=pos_edge_index, edge_attr=edge_attr)

def cast_data_vector_RH(dataset):
    """
        convert subject vectors to graph and append it in a list
    """

    dataset_g = []

    for subj in range(dataset.shape[0]):
        dataset_g.append(convert_vector_to_graph_RH(dataset[subj]))

    return dataset_g
def cast_data_vector_HHR(dataset):
    """
        convert subject vectors to graph and append it in a list
    """

    dataset_g = []

    for subj in range(dataset.shape[0]):
        dataset_g.append(convert_vector_to_graph_HHR(dataset[subj]))

    return dataset_g

def cast_data_vector_FC(dataset):
    """
        convert subject vectors to graph and append it in a list
    """

    dataset_g = []

    for subj in range(dataset.shape[0]):
        dataset_g.append(convert_vector_to_graph_FC(dataset[subj]))

    return dataset_g
def convert_generated_to_graph_268(data1):
    """
        convert generated output from G to a graph
    """

    dataset = []

    for data in data1:
        counter = 0
        N_ROI = 268
        pos_edge_index = torch.zeros(2, N_ROI * N_ROI, dtype=torch.long)
        for i in range(N_ROI):
            for j in range(N_ROI):
                pos_edge_index[:, counter] = torch.tensor([i, j])
                counter += 1

        x = data
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
        data = Data(x=x, pos_edge_index= pos_edge_index, edge_attr=data.view(71824, 1))
        dataset.append(data)

    return dataset
def convert_generated_to_graph(data):
    """
        convert generated output from G to a graph
    """

    dataset = []

# for data in data1:
    counter = 0
    N_ROI = N_TARGET_NODES
    pos_edge_index = torch.zeros(2, N_ROI * N_ROI, dtype=torch.long)
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

    x = data
    pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
    data = Data(x=x, pos_edge_index= pos_edge_index, edge_attr=data.view(N_TARGET_NODES**2, 1))
    dataset.append(data)

    return dataset

def convert_generated_to_graph_Al(data1):
    """
        convert generated output from G to a graph
    """

    dataset = []

    # for data in data1:
    counter = 0
    N_ROI = N_SOURCE_NODES
    pos_edge_index = torch.zeros(2, N_ROI * N_ROI, dtype=torch.long)
    for i in range(N_ROI):
        for j in range(N_ROI):
            pos_edge_index[:, counter] = torch.tensor([i, j])
            counter += 1

    # x = data
    pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
    data = Data(x=data1, pos_edge_index=pos_edge_index, edge_attr=data1.view(N_SOURCE_NODES*N_SOURCE_NODES, 1))
    dataset.append(data)

    return dataset