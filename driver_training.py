import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("core/")

import matplotlib.pyplot as plt
import io_utils as io
import nn_utils as nn
from geometry import Geometry
from resampled_geometry import ResampledGeometry
from data_container import DataContainer
import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def main():
    model = 'data/3d_flow_ini_1d_quad/0111_0001.vtp'
    soln = io.read_geo(model).GetOutput()
    soln_array, _, p_array = io.get_all_arrays(soln)
    pressures, velocities = io.gather_pressures_velocities(soln_array)
    geometry = Geometry(p_array)
    # geometry.plot()
    rgeo = ResampledGeometry(geometry, 5)
    # # rgeo.plot()
    rgeo.compare_field_along_centerlines(velocities[700])
    # stencil_size = 5
    # data = DataContainer(rgeo, stencil_size, pressures, velocities, soln_array['area'])
    # nn.train_and_save_all_networks(data, stencil_size)
    # geometry.plot(field = soln_array['velocity_0.01400'])
    # rgeo.plot(field = soln_array['velocity_0.01400'])
    plt.show()

model = 'data/3d_flow_ini_1d_quad/0111_0001.vtp'
model = 'data/3d_flow_repository(old)/0111_0001_recomputed.vtp'
soln = io.read_geo(model).GetOutput()
soln_array, _, p_array = io.get_all_arrays(soln)
pressures, velocities = io.gather_pressures_velocities(soln_array)
geometry = Geometry(p_array)
rgeo = ResampledGeometry(geometry, 20)

#%%

nodes, edges, lengths = rgeo.generate_nodes()
# append zero lengths for autoloops
for i in range(0, nodes.shape[0]):
    lengths.append(0)
times = [t for t in pressures]
times.sort()
gpressures, gvelocities, areas = rgeo.generate_fields(pressures, velocities, soln_array['area'])

def compute_seq_diff(fields, times):
    diffs = {}
    for itime in range(len(times) - 1):
        diffs[times[itime]] = (fields[times[itime + 1]] - fields[times[itime]])
        
    return diffs

def normalize(field):
    m = np.min(field)
    M = np.max(field)
    return (field - m) / (M - m), m, M

def normalize_sequence(fields):
    m = np.inf
    M = 0
    for t in fields:
        m = np.min((m, np.min(fields[t])))
        M = np.max((M, np.max(fields[t])))
        
    for t in fields:
        fields[t] = (fields[t] - m) / (M - m)
        
    return fields, m, M

diffp = compute_seq_diff(gpressures, times)
diffp, mdp, Mdp = normalize_sequence(diffp)

diffq = compute_seq_diff(gvelocities, times)
diffq, mdq, Mdq = normalize_sequence(diffq)

gpressures, mp, Mp = normalize_sequence(gpressures)
gvelocities, mq, Mq = normalize_sequence(gvelocities)
areas, ma, Ma = normalize(areas)

for jnodes in range(nodes.shape[1]):
    aa, m_, M_ = normalize(nodes[:,jnodes])
    nodes[:,jnodes] = aa

class AortaDataset(DGLDataset):
    def __init__(self, nodes, edges, lengths, gpressures, gvelocities, areas, times, diffp, diffq):
        self.nodes = nodes
        self.edges = edges
        self.lengths = lengths
        self.gpressures = gpressures
        self.gvelocities = gvelocities
        self.areas = areas
        self.times = times
        self.diffp = diffp
        self.diffq = diffq
        super().__init__(name='aorta')

    def process(self):
        self.graphs = []
        self.labels = []
        nnodes = self.nodes.shape[0]
        ntrain = int(nnodes * 0.6)
        nval = int(nnodes * 0.2)
        for itime in range(0, len(self.times) - 1):
            print(itime)
            g = dgl.graph((self.edges[:,0], self.edges[:,1]))
            # we add a self loop because we want the prediction in each node
            # to depend on itself
            g = dgl.to_bidirected(g)
            g = dgl.add_self_loop(g)
            features = gpressures[times[itime]]
            features = np.hstack((features, gvelocities[times[itime]]))
            features = np.hstack((features, areas, self.nodes))
            g.ndata['features'] = torch.from_numpy(features)
            # g.edata['length'] = torch.from_numpy(np.array(lengths))
            # label = gpressures[times[itime+1]]-gpressures[times[itime]]
            # label = np.hstack((label, gvelocities[times[itime+1]]-gvelocities[times[itime]]))
            # label = gpressures[times[itime+1]]
            # label = np.hstack((label, gvelocities[times[itime+1]]))
            label = self.diffp[times[itime]]
            label = np.hstack((label, self.diffq[times[itime]]))
            self.labels.append(label)
            train_mask = torch.zeros(nnodes, dtype=torch.bool)
            val_mask = torch.zeros(nnodes, dtype=torch.bool)
            test_mask = torch.zeros(nnodes, dtype=torch.bool)
            train_mask[:ntrain] = True
            val_mask[ntrain:ntrain + nval] = True
            test_mask[ntrain + nval:] = True
            g.ndata['train_mask'] = train_mask
            g.ndata['val_mask'] = val_mask
            g.ndata['test_mask'] = test_mask
            self.graphs.append(g)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

#%%

dataset = AortaDataset(nodes, edges, lengths, gpressures, gvelocities, areas, times, diffp, diffq)

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False)

from torch.nn.modules.module import Module
from dgl.nn import ChebConv
from dgl.nn import GraphConv
import torch.nn.functional as F

# class GCN(Module):
#     def __init__(self, in_feats, h_feats):
#         super(GCN, self).__init__()
#         # lm = dgl.laplacian_lambda_max(g)
#         # self.conv1 = ChebConv(in_feats, h_feats, 2, bias = True)
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, 2)
#         self.double()
#         self.lms = {}

#     def forward(self, g, in_feat):
#         # if g not in self.lms:
#         #     self.lms[g] = dgl.laplacian_lambda_max(g)
#         # h = self.conv1(g, in_feat, lambda_max = self.lms[g])
#         h = self.conv1(g, in_feat)
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         g.ndata['h'] = h
#         return h
    
class GCN(Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        # lm = dgl.laplacian_lambda_max(g)
        self.conv1 = ChebConv(in_feats, h_feats, 3, bias = True)
        # self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, 2)
        self.double()
        self.lms = {}

    def forward(self, g, in_feat):
        if g.num_nodes() not in self.lms:
            print('here')
            self.lms[g.num_nodes()] = dgl.laplacian_lambda_max(g)
        h = self.conv1(g, in_feat, lambda_max = self.lms[g.num_nodes()])
        # h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return h


# Create the model with given dimensions
ls_size = 32
infeat = 6
model = GCN(6, ls_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['features'].double())
        loss = F.mse_loss(pred, torch.reshape(labels, pred.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('ep = ' + str(epoch) + ' loss = ' + str(loss.detach().numpy()))

#%%

# # Recover the original graph elements from the minibatch
# for batched_graph, labels in train_dataloader:
#     graphs = dgl.unbatch(batched_graph)
#     for igraph in range(len(graphs)):
#         graph = graphs[igraph]
#         pred = model(graph, graph.ndata['features'].double())
#         true = labels[igraph,:,:]
#         fig1 = plt.figure()
#         ax1 = plt.axes()
#         deltap = mdp + pred[:,0].detach().numpy() * (Mdp - mdp)
#         predp = deltap + mp + graph.ndata['features'].detach().numpy()[:,0] * (Mp - mp)
#         truedeltap =  mdp + true[:,0].detach().numpy() * (Mdp - mdp)
#         truep = truedeltap + mp + graph.ndata['features'].detach().numpy()[:,0] * (Mp - mp)
#         ax1.plot(predp,'r')
#         ax1.plot(truep,'--b')
#         ax1.set_title('pressure')
    
#         fig2 = plt.figure()
#         ax2 = plt.axes()
#         deltaq = mdq + pred[:,1].detach().numpy() * (Mdq - mdq)
#         predq = deltaq + mq + graph.ndata['features'].detach().numpy()[:,1] * (Mq - mq)
#         truedeltaq =  mdq + true[:,1].detach().numpy() * (Mdq - mdq)
#         trueq = truedeltaq + mq + graph.ndata['features'].detach().numpy()[:,1] * (Mq - mq)
#         ax2.plot(predq,'r')
#         ax2.plot(trueq,'--b')
#         ax2.set_title('velocity')
it = iter(train_dataloader)
batch = next(it)
batched_graph, labels = batch
graph = dgl.unbatch(batched_graph)[0]
for it in range(0, len(times) - 1):
    t = times[it]
    gp = gpressures[t]
    gq = gvelocities[t]
    features = torch.from_numpy(np.hstack((gp,gq,areas, nodes)))
    pred = model(graph,features)
    
    # fig1 = plt.figure()
    # ax1 = plt.axes()
    # deltap = mdp + pred[:,0].detach().numpy() * (Mdp - mdp)
    # predp = deltap + mp + gpressures[t].squeeze() * (Mp - mp)
    # truedeltap =  mdp + true[:,0].detach().numpy() * (Mdp - mdp)
    # truep =  mp + gpressures[t] * (Mp - mp)
    # ax1.plot(predp / 1333.2,'r')
    # ax1.plot(truep / 1333.2,'--b')
    # ax1.set_title('pressure t = ' + str(t))
    # ax1.set_ylim((mp / 1333.2,Mp / 1333.2))
    # ax1.set_xlim((0, nodes.shape[0]))
    
    fig2 = plt.figure()
    ax2 = plt.axes()
    deltaq = mdq + pred[:,1].detach().numpy() * (Mdq - mdq)
    predq = deltaq + mq + gvelocities[t].squeeze() * (Mq - mq)
    trueq = mq + gvelocities[times[it + 1]] * (Mq - mq)
    ax2.plot(predq,'r')
    ax2.plot(trueq,'--b')
    ax2.set_title('velocity t = ' + str(t))
    ax2.set_ylim((mq,Mq))
    ax2.set_xlim((0, nodes.shape[0]))
    
#%%

it = iter(train_dataloader)
batch = next(it)
batched_graph, labels = batch
graph = dgl.unbatch(batched_graph)[0]
    
tin = 2
gp = gpressures[times[tin]]
gq = gvelocities[times[tin]]
for it in range(tin, len(times)-1):
    t = times[it]
    features = torch.from_numpy(np.hstack((gp,gq,areas, nodes)))
    pred = model(graph,features)
    
    
    deltap = mdp + pred[:,0].detach().numpy() * (Mdp - mdp)
    predp = np.expand_dims(deltap, axis = 1) + mp + gp * (Mp - mp)
    truep =  mp + gpressures[times[it + 1]] * (Mp - mp)
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(predp / 1333.2,'r')
    ax1.plot(truep / 1333.2,'--b')
    ax1.set_title('pressure t = ' + str(t))
    ax1.set_ylim((mp / 1333.2,Mp / 1333.2))
    ax1.set_xlim((0, nodes.shape[0]))
    
    deltaq = mdq + pred[:,1].detach().numpy() * (Mdq - mdq)
    predq = np.expand_dims(deltaq, axis = 1) + mq + gq * (Mq - mq)
    trueq = mq + gvelocities[times[it + 1]] * (Mq - mq)
    # fig2 = plt.figure()
    # ax2 = plt.axes()
    # ax2.plot(predq,'r')
    # ax2.plot(trueq,'--b')
    # ax2.set_title('velocity t = ' + str(t))
    # # ax2.set_ylim((mq,Mq))
    # ax2.set_xlim((0, nodes.shape[0]))
    
    if it < 13:
        gp = gpressures[times[it+1]]
        gq = gvelocities[times[it+1]]
    else:
        gp = (predp - mp) / (Mp - mp)
        gq = (predq - mq) / (Mq - mq)
    

