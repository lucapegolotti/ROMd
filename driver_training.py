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


model = 'data/3d_flow_ini_1d_quad/0111_0001.vtp'
model = 'data/3d_flow_repository(old)/0111_0001_recomputed.vtp'
model = 'data/output_no_sphere.vtp'
soln = io.read_geo(model).GetOutput()
soln_array, _, p_array = io.get_all_arrays(soln)

pressures, velocities = io.gather_pressures_velocities(soln_array)
geometry = Geometry(p_array)
rgeo = ResampledGeometry(geometry, 10)

#%%

nodes, edges, lengths, inlet_node, outlet_nodes = rgeo.generate_nodes()

# append zero lengths for autoloops
for i in range(0, nodes.shape[0]):
    lengths.append(0)
times = [t for t in pressures]
times.sort()
times = times[10:]
gpressures, gvelocities, areas = rgeo.generate_fields(pressures, velocities, soln_array['area'])

def compute_seq_diff(fields, times):
    diffs = {}
    
    for itime in range(0,len(times)-1):
        diffs[times[itime]] = (fields[times[itime+1]] - fields[times[itime]])
        
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
        print('---')
        print(times[itime])
        print(np.max(np.abs(fields[times[itime]])))
        
    return fields, m, M

def standardize(field):
    m = np.mean(field)
    stdv = np.std(field)
    return (field - m) / stdv, m, stdv

def standardize_sequence(fields):
    
    allfields = np.zeros((0,1))
    for t in fields:
        allfields = np.vstack((allfields,fields[t]))
    
    m = np.mean(allfields)
    stdv = np.std(allfields)
    
    for t in fields:
        fields[t] = (fields[t] - m) / stdv
        
    return fields, m, stdv

diffp = compute_seq_diff(gpressures, times)
diffp, mdp, stdvdp = standardize_sequence(diffp)

diffq = compute_seq_diff(gvelocities, times)
diffq, mdq, stdvdq = standardize_sequence(diffq)

gpressures, mp, stdp = standardize_sequence(gpressures)
gvelocities, mq, stdq = standardize_sequence(gvelocities)
areas, ma, stda = standardize(areas)

g = dgl.graph((edges[:,0], edges[:,1]))
g = dgl.to_bidirected(g)
# g = dgl.add_self_loop(g) 

nnodes = nodes.shape[0]
edge_features = []

edg0 = g.edges()[0]
edg1 = g.edges()[1]
N = edg0.shape[0]
for j in range(0, N):
    diff = nodes[edg1[j],:] - nodes[edg0[j],:]
    diff = np.hstack((diff, np.linalg.norm(diff)))
    edge_features.append(diff)
    
edge_features = np.array(edge_features)
for j in range(4):
    normf, _, _ = standardize(edge_features[:,j])
    edge_features[:,j] = normf
    
node_degree = []
for j in range(0, nnodes):
    node_degree.append(np.count_nonzero(edg0 == j) + np.count_nonzero(edg1 == j))

node_degree = np.array(node_degree)
# node_degree[inlet_node] = 0
# node_degree[inlet_node] = 1
degrees = set()
for j in range(0, nnodes):
    degrees.add(node_degree[j])

one_hot_degrees = node_degree * 0
# we use 0 for inlet and 1 for outlet
count = 2
for degree in degrees:
    if degree != 1:
        indx = np.where(node_degree == degree)[0]
        one_hot_degrees[indx] = count
        count = count + 1
        
one_hot_degrees[inlet_node] = 0
one_hot_degrees[outlet_nodes] = 1

# node_degree = one_hot_degrees
# m = np.min(node_degree)
# M = np.max(node_degree)
# node_degree = (node_degree - m) / (M - m)
node_degree = np.expand_dims(node_degree, axis = 1).astype(np.float32)

#%%

class AortaDataset(DGLDataset):
    def __init__(self, nodes, edges, lengths, gpressures, gvelocities, areas, times, diffp, diffq, edge_features, node_degree):
        self.nodes = nodes
        self.edges = edges
        self.lengths = lengths
        self.gpressures = gpressures
        self.gvelocities = gvelocities
        self.areas = areas
        self.times = times
        self.diffp = diffp
        self.diffq = diffq
        self.edge_features = edge_features
        self.node_degree = node_degree
        super().__init__(name='aorta')

    def process(self):
        self.graphs = []
        self.labels = []
        nnodes = self.nodes.shape[0]
        ntrain = int(nnodes * 0.6)
        nval = int(nnodes * 0.2)
        
        edge_features = []
        node_degree = []
        enrich = 3
        rate_noise = 0.01
        for itime in range(0, len(self.times) - 1):
            
            for i in range(enrich):
                g = dgl.graph((self.edges[:,0], self.edges[:,1]))
                # we add a self loop because we want the prediction in each node
                # to depend on itself
                g = dgl.to_bidirected(g)
                # g = dgl.add_self_loop(g)
                        
                features = np.hstack((gpressures[times[itime]], gvelocities[times[itime]]))
                features[inlet_node,1] = gvelocities[times[itime + 1]][inlet_node].squeeze()
                features[outlet_nodes,0] = gpressures[times[itime + 1]][outlet_nodes].squeeze()
                noise = 0
                features = np.hstack((features, areas, self.node_degree))
                # if i != 0:
                noise = np.random.normal(0, rate_noise, (features.shape[0], 2)) * np.abs(features[:,:2])
                features[:,:2] = noise.astype(np.float64) + features[:,:2]
                g.ndata['features'] = torch.from_numpy(features)
                g.edata['dist'] = torch.from_numpy(self.edge_features)
                label = self.diffp[times[itime]]
                label = np.hstack((label, self.diffq[times[itime]]))
                label = label - noise
                # label[inlet_node,1] = 0
                # label[outlet_nodes,0] = 0
                self.labels.append(label)
                if (np.max(np.abs(label[:,0])) > 10):
                    print('----')
                    print(times[itime])
                    print(np.max(np.abs(label[:,0])))
                g.ndata['labels'] = torch.from_numpy(label)
                g.ndata['time'] = torch.from_numpy(np.ones(features.shape[0]) * times[itime])
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

dataset = AortaDataset(nodes, edges, lengths, gpressures, gvelocities, areas, times, diffp, diffq, edge_features, node_degree)

#%%

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=10, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=10, drop_last=False)

from torch.nn.modules.module import Module
from dgl.nn import ChebConv
from dgl.nn import GraphConv
from dgl.nn import GATConv
from dgl.nn import RelGraphConv
from torch.nn import Linear
import torch.nn.functional as F
import dgllife
import dgl.function as fn

# class GCN(Module):
#     def __init__(self, in_feats, h_feats):
#         super(GCN, self).__init__()
#         # lm = dgl.laplacian_lambda_max(g)
#         # self.conv1 = ChebConv(in_feats, h_feats, 3, bias = False)
#         # self.conv1 = GATConv(h_feats, h_feats, num_heads = 1, feat_drop = 0.1)
#         self.num_heads = 2
#         self.conv1 = GATConv(in_feats, h_feats, num_heads = self.num_heads, feat_drop = 0.5)
#         self.conv2 = GATConv(h_feats, 2, num_heads = 1, feat_drop = 0.5)
#         # self.conv3 = GraphConv(h_feats, 2)
#         self.double()
#         self.lms = {}

#     def forward(self, g, in_feat):
#         if g.num_nodes() not in self.lms:
#             self.lms[g.num_nodes()] = dgl.laplacian_lambda_max(g)
#         # h = self.conv1(g, in_feat, lambda_max = self.lms[g.num_nodes()])
#         h = self.conv1(g, in_feat)
#         h = torch.sum(h, dim = 1)
#         h = h / self.num_heads
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         # h = F.relu(h)
#         # h = self.conv3(g, h)
#         # g.ndata['h'] = h
#         return h
        

class GraphNet(Module):
    def __init__(self, in_feats_nodes, in_feats_edges, latent, h_feats, L, inlet_node, outlet_nodes):
        super(GraphNet, self).__init__()
        self.encoder_nodes = Linear(in_feats_nodes, latent).double()
        self.encoder_edges = Linear(in_feats_edges, latent).double()
        self.processor_edges = []
        self.processor_nodes = []
        for i in range(L):      
            self.processor_edges.append(Linear(latent * 3, latent).double())
            self.processor_nodes.append(Linear(latent * 2, latent).double())
        self.L = L
        self.output = Linear(latent, h_feats).double()
        self.inlet_node = inlet_node
        self.outlet_nodes = outlet_nodes
        
    def encode_nodes(self, nodes):
        f = nodes.data['features_c']
        enc_features = self.encoder_nodes(f)
        enc_features = F.relu(enc_features)
        return {'proc_node': enc_features}
    
    def encode_edges(self, edges):
        f = edges.data['dist']
        enc_features = self.encoder_edges(f)
        enc_features = F.relu(enc_features)
        return {'proc_edge': enc_features}
    
    def process_edges(self, edges, layer):
        f1 = edges.data['proc_edge']
        f2 = edges.src['proc_node']
        f3 = edges.dst['proc_node']
        proc_edge = self.processor_edges[layer](torch.cat((f1, f2, f3),dim=1))
        proc_edge = F.relu(proc_edge)
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge' : proc_edge}
    
    def process_nodes(self, nodes, layer):
        f1 = nodes.data['proc_node']
        f2 = nodes.data['pe_sum']
        proc_node = self.processor_nodes[layer](torch.cat((f1, f2),dim=1))
        proc_node = F.relu(proc_node)
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node' : proc_node}
    
    def decode(self, nodes):
        f = nodes.data['proc_node']
        h = self.output(f)
        return {'h' : h}
    
    def forward(self, g, in_feat):
        g.ndata['features_c'] = in_feat
        g.apply_nodes(self.encode_nodes)
        g.apply_edges(self.encode_edges)
        for i in range(self.L):
            def pe(edges):
                return self.process_edges(edges, i)
            def pn(nodes):
                return self.process_nodes(nodes, i)
            g.apply_edges(pe)
            # aggregate new edge features in nodes
            g.update_all(fn.copy_e('proc_edge', 'm'), fn.sum('m', 'pe_sum'))
            g.apply_nodes(pn)
        g.apply_nodes(self.decode)
        
        # # averaging?
        # for i in range(5):
        #     g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        
        g.ndata['h'][self.inlet_node,1] = 0
        g.ndata['h'][self.outlet_nodes,0] = 0
        return g.ndata['h']
    
def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

# Create the model with given dimensions
latent = 16
infeat_nodes = 4
infeat_edges = 4
model = GraphNet(infeat_nodes, infeat_edges, latent, 2, 10, inlet_node, outlet_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
nepochs = 1000
for epoch in range(nepochs):
    print('ep = ' + str(epoch))
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['features'].double())
        # loss = F.mse_loss(pred, torch.reshape(labels, pred.shape))
        weight = torch.ones(pred.shape) 
        weight[inlet_node,0] = 10
        weight[outlet_nodes,1] = 10
        weight[inlet_node,1] = 0
        weight[outlet_nodes,0] = 0
        loss = weighted_mse_loss(pred, torch.reshape(labels, pred.shape), weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    fig1 = plt.figure()
    ax1 = plt.axes() 
    ax1.plot(pred[:,0].detach().numpy(),'-r')
    ax1.plot(torch.reshape(labels, pred.shape).detach().numpy()[:,0],'-b')
    ax1.set_ylim(np.min(torch.reshape(labels, pred.shape).detach().numpy()[:,0]),
                 np.max(torch.reshape(labels, pred.shape).detach().numpy()[:,0]))
    ax1.set_title(str(batched_graph.ndata['time'][0].numpy()))
    plt.show()
        
    print('\tloss = ' + str(loss.detach().numpy()))

#%%

def find_solutions(model, graph, guess, prev, pnodes, qnodes, bcs_p, bcs_q):

    def residual(x):
        pred = model(graph, x)
        # res = pred + prev[:,0:2] - x[:,0:2]
        res = pred - x[:,0:2]
        res = torch.hstack((res, torch.zeros(x.shape[0], x.shape[1] - 2)))
        res[pnodes,0] = 0
        res[qnodes,1] = 0
        return res
    
    def jac(x):
        J = torch.autograd.functional.jacobian(residual, x)
        J[pnodes,0,:,0] = 0
        J[pnodes,0,:,1] = 0
        J[pnodes,0,pnodes,0] = 1
        J[qnodes,1,:,0] = 0
        J[qnodes,1,:,1] = 0
        J[qnodes,1,qnodes,1] = 1
        return J
        
    N = 10
    tol = 1e-4
    x = guess
    x[pnodes,0] = bcs_p
    x[qnodes,1] = bcs_q
    res = residual(x)
    inerr = torch.norm(res)
    err = inerr
    count = 0
    while err / inerr > tol:
        print('it = ' + str(count) + ' rel err = ' + str((err / inerr).detach().numpy()) + ' abs err = ' + str(err.detach().numpy()))
        J = jac(x)
        dx = torch.zeros(res.shape)
        globalJ1 = torch.hstack((J[:,0,:,0], J[:,0,:,1]))
        globalJ2 = torch.hstack((J[:,1,:,0], J[:,1,:,1]))
        globalJ = torch.vstack((globalJ1, globalJ2))
        # print(torch.linalg.cond(globalJ))
        print(globalJ[:10,:10])
        R = torch.hstack((res[:,0],res[:,1])) # torch.reshape(res[:,0:2], (res.shape[0] * 2, 1))
        dx = torch.transpose(torch.reshape(torch.linalg.solve(globalJ, R), (2, res.shape[0])),0,1)
        # dx = torch.vstack((dx[:res.shape[0]],dx[res.shape[0]:]))
        
        # in principle it should be like this but the jacobians of the cross
        # terms are singular
        # for i in range(0,2):
        #     for j in range(0,2):
        #         print(torch.linalg.cond(J[:,i,:,j]))
        #         dx[:,i] = dx[:,i] + torch.linalg.solve(J[:,i,:,j], res[:,j])
        # dx = torch.zeros(res[:,0:2].shape)
        # for i in range(0,2):
        #     dx[:,i] = dx[:,i] + torch.linalg.solve(J[:,i,:,i], res[:,i])
        x[:,0:2] = x[:,0:2] - dx
        res = residual(x)
        err = torch.norm(res)
        count = count + 1
    print('done ' + str(count) + ' rel err = ' + str((err / inerr).detach().numpy()) + ' abs err = ' + str(err.detach().numpy()))
    return x
        
#%%


# it = iter(train_dataloader)
# batch = next(it)
# batched_graph, labels = batch
# graph = dgl.unbatch(batched_graph)[0]
# count = 0
# for it in range(0, len(times) - 1):
#     t = times[it]
#     gp = gpressures[t]
#     gq = gvelocities[t]
#     features = torch.from_numpy(np.hstack((gp,gq,areas, nodes)))
    
    
#     pred = find_solutions(model, gp,)
#     pred = model(graph,features)
    
#     fig1 = plt.figure()
#     ax1 = plt.axes()
#     deltap = mdp + pred[:,0].detach().numpy() * (Mdp - mdp)
#     predp = deltap + mp + gpressures[t].squeeze() * (Mp - mp)
#     truep =  mp + gpressures[t] * (Mp - mp)
#     ax1.plot(predp / 1333.2,'r')
#     ax1.plot(truep / 1333.2,'--b')
#     ax1.set_title('pressure t = ' + str(t))
#     ax1.set_ylim((mp / 1333.2,Mp / 1333.2))
#     ax1.set_xlim((0, nodes.shape[0]))
#     plt.savefig('pressure/t' + str(count).rjust(3, '0'))

    
#     fig2 = plt.figure()
#     ax2 = plt.axes()
#     deltaq = mdq + pred[:,1].detach().numpy() * (Mdq - mdq)
#     predq = deltaq + mq + gvelocities[t].squeeze() * (Mq - mq)
#     trueq = mq + gvelocities[times[it + 1]] * (Mq - mq)
#     ax2.plot(predq,'r')
#     ax2.plot(trueq,'--b')
#     ax2.set_title('flowrate t = ' + str(t))
#     ax2.set_ylim((mq,Mq))
#     ax2.set_xlim((0, nodes.shape[0]))
#     plt.savefig('flowrate/t' + str(count).rjust(3, '0'))
#     count = count + 1
    
#%% when we predict delta

it = iter(train_dataloader)
batch = next(it)
batched_graph, labels = batch
graph = dgl.unbatch(batched_graph)[0]
timeg = float(graph.ndata['time'][0])
nodes_degree = np.expand_dims(graph.ndata['features'][:,-1],1)
    
tin = 20
count = 0
gp = gpressures[times[tin]]
gq = gvelocities[times[tin]]
for it in range(tin, len(times)-1):
    t = times[it]
    nt = times[it + 1]
    tp = gpressures[nt]
    tq = gvelocities[nt]
    
    bcs_p = torch.Tensor(np.squeeze(tp[outlet_nodes]).astype(np.float64))
    bcs_q = torch.Tensor(np.squeeze(tq[inlet_node]).astype(np.float64))
    # pred = find_solutions(model, graph, prev, prev, outlet_nodes, inlet_node, bcs_p, bcs_q)
    
    # prev = torch.from_numpy(np.hstack((gp, gq, gp, gq, areas, nodes)))
    # pred = find_solutions(model, graph, prev, prev)
    prev = torch.from_numpy(np.hstack((gp, gq, areas, node_degree)))
    prev[inlet_node,1] = torch.from_numpy(tq[inlet_node].squeeze())
    prev[outlet_nodes,0] = torch.from_numpy(tp[outlet_nodes].squeeze())

    # prev = torch.from_numpy(np.hstack((tp, tq, gp, gq, areas, nodes_degree)))
    pred = model(graph, prev)
    # pred = find_solutions(model, graph, prev, prev, outlet_nodes, inlet_node, bcs_p, bcs_q)
    pred = pred.squeeze()
    deltap = mdp + pred[:,0].detach().numpy() * (stdvdp)
    predp = deltap + mp + prev[:,0].numpy() * (stdp)
    truep = mp + gpressures[nt] * stdp
    fig1 = plt.figure()
    ax1 = plt.axes() 
    ax1.plot(predp / 1333.2,'r')
    ax1.plot(truep / 1333.2,'-b')
    ax1.set_title('pressure t = ' + str(t))
    # ax1.set_ylim((mp / 1333.2,Mp / 1333.2))
    ax1.set_xlim((0, nodes.shape[0]))
    plt.savefig('pressure/t' + str(count).rjust(3, '0'))

    deltaq = mdq + pred[:,1].detach().numpy() * (stdvdq)
    predq = deltaq + mq + prev[:,1].numpy() * stdq
    trueq = mq + gvelocities[nt] * stdq
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(predq,'r')
    ax2.plot(trueq,'--b')
    ax2.set_title('velocity t = ' + str(t))
    # ax2.set_ylim((mq,Mq))
    ax2.set_xlim((0, nodes.shape[0]))
    plt.savefig('flowrate/t' + str(count).rjust(3, '0'))
    plt.show()
    # if it < 13:
    gp = gpressures[times[it+1]]
    gq = gvelocities[times[it+1]]
    gp = tp
    gq = tq
    # else:
    gp = np.expand_dims((predp - mp),1) / stdp
    gq = np.expand_dims((predq - mq),1) / stdq
    count = count + 1

#%% when we predict the function

it = iter(train_dataloader)
batch = next(it)
batched_graph, labels = batch
graph = dgl.unbatch(batched_graph)[0]
    
tin = 2
gp = gpressures[times[tin]]
gq = gvelocities[times[tin]]
count = 0
for it in range(tin, len(times)-1):
    tp = gpressures[times[it + 1]]
    tq = gvelocities[times[it + 1]]
    t = times[it]
    prev = torch.from_numpy(np.hstack((gp, gq, gp, gq, areas, nodes)))
    # prev = torch.from_numpy(np.hstack((gpressures[times[it + 1]], gvelocities[times[it + 1]], gp, gq, areas, nodes)))
    
    bcs_p = torch.Tensor(np.squeeze(tp[outlet_nodes]).astype(np.float64))
    bcs_q = torch.Tensor(np.squeeze(tq[inlet_node]).astype(np.float64))
    
    bcsp = gp * 0
    bcsp[outlet_nodes] = tp[outlet_nodes]
    bcsq = gq * 0
    bcsq[inlet_node] = tq[inlet_node]
    prev = torch.from_numpy(np.hstack((tp, tq, gp, gq, areas, node_degree)))
    # pred = model(graph, prev)
    pred = find_solutions(model, graph, prev, prev, outlet_nodes, inlet_node, bcs_p, bcs_q)

    predp = mp + pred[:,0].detach().numpy() * (Mp - mp)
    truep =  mp + gpressures[times[it + 1]] * (Mp - mp)
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(predp / 1333.2,'r')
    ax1.plot(truep / 1333.2,'--b')
    ax1.set_title('pressure t = ' + str(t))
    # ax1.set_ylim((mp / 1333.2,Mp / 1333.2))
    ax1.set_xlim((0, nodes.shape[0]))
    plt.savefig('pressure/t' + str(count).rjust(3, '0'))

    predq = mq + pred[:,1].detach().numpy() * (Mq - mq)
    trueq = mq + gvelocities[times[it + 1]] * (Mq - mq)
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(predq,'r')
    ax2.plot(trueq,'--b')
    ax2.set_title('velocity t = ' + str(t))
    # ax2.set_ylim((mq,Mq))
    ax2.set_xlim((0, nodes.shape[0]))
    plt.savefig('flowrate/t' + str(count).rjust(3, '0'))

    plt.show()

    # if it < 13:
    gp = gpressures[times[it+1]]
    gq = gvelocities[times[it+1]]
    # else:
    # gp = np.expand_dims((predp - mp) / (Mp - mp),1)
    # gq = np.expand_dims((predq - mq) / (Mq - mq),1)
    print(gp.shape)
    count = count + 1

