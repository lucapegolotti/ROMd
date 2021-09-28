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
soln_array, _, p_array = io.get_all_arrays(soln, 180)
pressures, velocities = io.gather_pressures_velocities(soln_array)
geometry = Geometry(p_array)
rgeo = ResampledGeometry(geometry, 20)

#%%

nodes, edges, lengths, inlet_node, outlet_nodes = rgeo.generate_nodes()
# append zero lengths for autoloops
for i in range(0, nodes.shape[0]):
    lengths.append(0)
times = [t for t in pressures]
times.sort()
gpressures, gvelocities, areas = rgeo.generate_fields(pressures, velocities, soln_array['area'])

def compute_seq_diff(fields, times):
    diffs = {}

    # for itime in range(1,len(times) - 1):
    #     diffs[times[itime]] = (fields[times[itime + 1]] - fields[times[itime-1]]) / 2
    
    for itime in range(1,len(times)):
        diffs[times[itime]] = (fields[times[itime]] - fields[times[itime-1]])
        
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
    
#%%

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
        
        enrich = 3
        rate_noise = 0.01
        for itime in range(0, len(self.times) - 1):
            
            for i in range(enrich):
                g = dgl.graph((self.edges[:,0], self.edges[:,1]))
                # we add a self loop because we want the prediction in each node
                # to depend on itself
                g = dgl.to_bidirected(g)
                g = dgl.add_self_loop(g)
                bcsp = gpressures[times[itime]] * 0
                bcsp[outlet_nodes] = gpressures[times[itime]][outlet_nodes]
                bcsq = gvelocities[times[itime]] * 0
                bcsq[inlet_node] = gpressures[times[itime]][inlet_node]
                features = np.hstack((bcsp,bcsq))
                features = np.hstack((features, gpressures[times[itime-1]]))
                features = np.hstack((features, gvelocities[times[itime-1]]))
                features = np.hstack((features, areas, self.nodes))
                if i != 0:
                    noise = np.random.rand(features.shape[0], features.shape[1])*rate_noise - rate_noise/2
                    featurs = features + noise
                g.ndata['features'] = torch.from_numpy(features)
                # g.edata['length'] = torch.from_numpy(np.array(lengths))
                # label = gpressures[times[itime+1]]-gpressures[times[itime]]
                # label = np.hstack((gpressures[times[itime]],
                #                    gvelocities[times[itime]]))
                # label = np.hstack((label, gvelocities[times[itime+1]]-gvelocities[times[itime]]))
                # label = gpressures[times[itime+1]]
                # label = np.hstack((label, gvelocities[times[itime+1]]))
                label = self.diffp[times[itime]]
                label = np.hstack((label, self.diffq[times[itime]]))
                self.labels.append(label)
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

dataset = AortaDataset(nodes, edges, lengths, gpressures, gvelocities, areas, times, diffp, diffq)

#%%

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
from dgl.nn import GATConv
import torch.nn.functional as F

class GCN(Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        # lm = dgl.laplacian_lambda_max(g)
        self.conv1 = ChebConv(in_feats, 2, 3, bias = False)
        # self.conv1 = GATConv(h_feats, h_feats, num_heads = 1, feat_drop = 0.1)
        # self.conv1 = GraphConv(in_feats, h_feats)
        # self.conv2 = GraphConv(h_feats, 2)
        # self.conv3 = GraphConv(h_feats, 2)
        self.double()
        self.lms = {}

    def forward(self, g, in_feat):
        if g.num_nodes() not in self.lms:
            self.lms[g.num_nodes()] = dgl.laplacian_lambda_max(g)
        h = self.conv1(g, in_feat, lambda_max = self.lms[g.num_nodes()])
        # h = self.conv1(g, in_feat)
        # h = F.relu(h)
        # h = self.conv2(g, h)
        # h = F.relu(h)
        # h = self.conv3(g, h)
        # g.ndata['h'] = h
        return h

# Create the model with given dimensions
ls_size = 8
infeat = 8
model = GCN(infeat, ls_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
nepochs = 1000
for epoch in range(nepochs):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['features'].double())
        loss = F.mse_loss(pred, torch.reshape(labels, pred.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('ep = ' + str(epoch) + ' loss = ' + str(loss.detach().numpy()))

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
        print(x[:,1])
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
    
tin = 2
gp = gpressures[times[tin]]
gq = gvelocities[times[tin]]
count = 0
for it in range(tin, len(times)-1):
    t = times[it]
    tp = gpressures[times[it + 1]]
    tq = gvelocities[times[it + 1]]
    
    bcs_p = torch.Tensor(np.squeeze(tp[outlet_nodes]).astype(np.float64)).double()
    bcs_q = torch.Tensor(np.squeeze(tq[inlet_node]).astype(np.float64)).double()
    # pred = find_solutions(model, graph, prev, prev, outlet_nodes, inlet_node, bcs_p, bcs_q)
    bcsp = gp * 0
    bcsp[outlet_nodes] = tp[outlet_nodes]
    bcsq = gq * 0
    bcsq[inlet_node] = tq[inlet_node]
    
    # prev = torch.from_numpy(np.hstack((gp, gq, gp, gq, areas, nodes)))
    # pred = find_solutions(model, graph, prev, prev)
    
    prev = torch.from_numpy(np.hstack((bcsp, bcsq, gp, gq, areas, nodes)))
    pred = model(graph, prev)
    
    deltap = mdp + pred[:,0].detach().numpy() * (Mdp - mdp)
    predp = np.expand_dims(deltap, axis = 1) + mp + gp * (Mp - mp)
    truep = mp + gpressures[times[it + 1]] * (Mp - mp)
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(predp / 1333.2,'r')
    ax1.plot(truep / 1333.2,'--b')
    ax1.set_title('pressure t = ' + str(t))
    ax1.set_ylim((mp / 1333.2,Mp / 1333.2))
    ax1.set_xlim((0, nodes.shape[0]))
    plt.savefig('pressure/t' + str(count).rjust(3, '0'))

    deltaq = mdq + pred[:,1].detach().numpy() * (Mdq - mdq)
    predq = np.expand_dims(deltaq, axis = 1) + mq + gq * (Mq - mq)
    trueq = mq + gvelocities[times[it + 1]] * (Mq - mq)
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(predq,'r')
    ax2.plot(trueq,'--b')
    ax2.set_title('velocity t = ' + str(t))
    ax2.set_ylim((mq,Mq))
    ax2.set_xlim((0, nodes.shape[0]))
    plt.savefig('flowrate/t' + str(count).rjust(3, '0'))

    # if it < 13:
    # gp = gpressures[times[it+1]]
    # gq = gvelocities[times[it+1]]
    # else:
    gp = (predp - mp) / (Mp - mp)
    gq = (predq - mq) / (Mq - mq)
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
    
    bcs_p = torch.Tensor(np.squeeze(tp[outlet_nodes]).astype(np.float64)).double()
    bcs_q = torch.Tensor(np.squeeze(tq[inlet_node]).astype(np.float64)).double()
    # pred = find_solutions(model, graph, prev, prev, outlet_nodes, inlet_node, bcs_p, bcs_q)
    bcsp = gp * 0
    bcsp[outlet_nodes] = tp[outlet_nodes]
    bcsq = gq * 0
    bcsq[inlet_node] = tq[inlet_node]
    prev = torch.from_numpy(np.hstack((bcsp, bcsq, gp, gq, areas, nodes)))
    pred = model(graph, prev)
    
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
    # gp = gpressures[times[it+1]]
    # gq = gvelocities[times[it+1]]
    # else:
    gp = np.expand_dims((predp - mp) / (Mp - mp),1)
    gq = np.expand_dims((predq - mq) / (Mq - mq),1)
    print(gp.shape)
    count = count + 1

