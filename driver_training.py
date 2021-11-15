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
model = 'data/output_sphere.vtp'
soln = io.read_geo(model).GetOutput()
soln_array, _, p_array = io.get_all_arrays(soln)

pressures, velocities = io.gather_pressures_velocities(soln_array)    

geometry = Geometry(p_array)

# for t in velocities:
#     velocities[t] = velocities[t] * soln_array['area']
    
for t in pressures:
    pressures[t] = pressures[t] / 1333.2

rgeo = ResampledGeometry(geometry, 10, remove_caps = True)

#%%

nodes, edges, lengths, inlet_node, outlet_nodes = rgeo.generate_nodes()

times = [t for t in pressures]
times.sort()
times = times[10:]
gpressures, gvelocities, areas = rgeo.generate_fields(pressures, velocities, soln_array['area'], times)

#%%
# plot pressures 

it = iter(train_dataloader)
    
tin = 0
count = 0
gp = gpressures[times[tin]]
gq = gvelocities[times[tin]]
for it in range(tin, len(times)-1):
    t = times[it]
   
    truep = gpressures[t]
    fig1 = plt.figure()
    ax1 = plt.axes() 
    ax1.plot(gpressures[times[it + 1]] - truep,'-b')
    ax1.set_title('pressure t = ' + str(t))
    plt.show()


#%% let's add noise using random walk as in https://arxiv.org/pdf/2002.09405.pdf

random_walks = 4
r_gpressures = []
r_gvelocities = []

for i in range(random_walks):
    new_gpressures = {}
    new_gvelocities = {}
    
    nP = gpressures[times[0]].shape[0]
    nQ = gvelocities[times[0]].shape[0]
    rate_noise = 0.00005
    noise_p = np.zeros((nP, 1))
    noise_q = np.zeros((nQ, 1))
    for itime in range(0,len(times)):
        gp = gpressures[times[itime]]
        noisep = noise_p + np.random.normal(0, rate_noise, (nP, 1)) * gp
        
        new_gpressures[times[itime]] = gpressures[times[itime]] + noisep
        
        gq = gvelocities[times[itime]]
        noiseq = noise_q + np.random.normal(0, rate_noise, (nQ, 1)) * gq
        
        new_gvelocities[times[itime]] = gvelocities[times[itime]] + noiseq
        
    r_gpressures.append(new_gpressures)
    r_gvelocities.append(new_gvelocities)
    
fig1 = plt.figure()
ax1 = plt.axes() 
ax1.plot(gpressures[times[itime]], '-r')
ax1.plot(new_gpressures[times[itime]],'--b')    

#%% compute diffs

def compute_seq_diff(fields, times):
    diffs = {}
    
    for itime in range(0,len(times)-1):
        diffs[times[itime]] = (fields[times[itime+1]] - fields[times[itime]])
        
    return diffs

diffp = compute_seq_diff(gpressures, times)
diffq = compute_seq_diff(gvelocities, times)

r_diffp = []
r_diffq = []
for i in range(random_walks):
    r_diffp.append(compute_seq_diff(r_gpressures[i], times))
    r_diffq.append(compute_seq_diff(r_gvelocities[i], times))


#%%

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


_normalize = False

if _normalize:
    diffp, mdp, Mdp = normalize_sequence(diffp)
    diffq, mdq, Mdq = normalize_sequence(diffq)
    
    gpressures, mp, Mp = normalize_sequence(gpressures)
    gvelocities, mq, Mq = normalize_sequence(gvelocities)
    areas, ma, stda = normalize(areas)
    
    for i in range(random_walks):
        r_diffp.append(compute_seq_diff(r_gpressures[i], times))
        r_diffq.append(compute_seq_diff(r_gvelocities[i], times))
        
        
        for t in r_gpressures[i]:
            if t != times[-1]:
                r_diffp[i][t] = (r_diffp[i][t] - mdp) / (Mdp - mdp)
                r_diffq[i][t] = (r_diffq[i][t] - mdq) / (Mdq - mdq)
            r_gpressures[i][t] = (r_gpressures[i][t] - mp) / (Mp - mp)
            r_gvelocities[i][t] = (r_gvelocities[i][t] - mq) / (Mq - mq)
    
    
else:
    
    diffp, mdp, stdvdp = standardize_sequence(diffp)
    diffq, mdq, stdvdq = standardize_sequence(diffq)
    
    gpressures, mp, stdp = standardize_sequence(gpressures)
    gvelocities, mq, stdq = standardize_sequence(gvelocities)
    areas, ma, stda = standardize(areas)
    
    for i in range(random_walks):
        r_diffp.append(compute_seq_diff(r_gpressures[i], times))
        r_diffq.append(compute_seq_diff(r_gvelocities[i], times))
        
        
        for t in r_gpressures[i]:
            if t != times[-1]:
                r_diffp[i][t] = (r_diffp[i][t] - mdp) / stdvdp
                r_diffq[i][t] = (r_diffq[i][t] - mdq) / stdvdq
            r_gpressures[i][t] = (r_gpressures[i][t] - mp) / stdp
            r_gvelocities[i][t] = (r_gvelocities[i][t] - mq) / stdq
    

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

fig1 = plt.figure()
ax1 = plt.axes(projection='3d') 

for i in range(edges.shape[0]):
    e1 = nodes[int(edges[i,0]),:]
    e2 = nodes[int(edges[i,1]),:]
    ax1.plot3D([e1[0],e2[0]], [e1[1],e2[1]], [e1[2],e2[2]], '-r')
    
for i in range(node_degree.size):
    if node_degree[i] > 4:
        ax1.scatter3D(nodes[i,0],nodes[i,1],nodes[i,2],c = 'black')
    
plt.show()

#%%

class AortaDataset(DGLDataset):
    def __init__(self, nodes, edges, lengths, gpressures, gvelocities, areas, times, diffp, diffq, edge_features, node_degree, r_gpressures, r_gvelocities, r_diffp, r_diffq):
        self.nodes = nodes
        self.edges = edges
        self.lengths = lengths
        self.allpressures = [gpressures] 
        self.allpressures = self.allpressures + r_gpressures
        self.allvelocities = [gvelocities]
        self.allvelocities = self.allvelocities + r_gvelocities
        self.areas = areas
        self.times = times
        self.alldiffp = [diffp]
        self.alldiffp = self.alldiffp + r_diffp
        self.alldiffq = [diffq]
        self.alldiffq = self.alldiffq + r_diffq
        self.edge_features = edge_features
        self.node_degree = node_degree
        super().__init__(name='aorta')

    def process(self):
        self.graphs = []
        self.labels = []
        nnodes = self.nodes.shape[0]
        ntrain = int(nnodes * 0.6)
        nval = int(nnodes * 0.2)
        
        nrollouts = len(self.allpressures)
        
        for i in range(nrollouts):
            edge_features = []
            node_degree = []
            for itime in range(0, len(self.times) - 1):
                g = dgl.graph((self.edges[:,0], self.edges[:,1]))
                g = dgl.to_bidirected(g)
                # g = dgl.add_self_loop(g)
                        
                features = np.hstack((self.allpressures[i][times[itime]], self.allvelocities[i][times[itime]]))
                # features[inlet_node,1] = self.allvelocities[i][times[itime + 1]][inlet_node].squeeze()
                # features[outlet_nodes,0] = self.allpressures[i][times[itime + 1]][outlet_nodes].squeeze()
                # features = np.hstack((self.alldiffp[i][times[itime]], self.alldiffq[i][times[itime]]))
                features = np.hstack((features, self.areas, self.node_degree))
                g.ndata['features'] = torch.from_numpy(features)
                g.edata['dist'] = torch.from_numpy(self.edge_features)
                label = self.alldiffp[i][times[itime]]
                label = np.hstack((label, self.alldiffq[i][times[itime]]))
                label = label
                self.labels.append(label)
                g.ndata['labels'] = torch.from_numpy(label)
                g.ndata['time'] = torch.from_numpy(np.ones(features.shape[0]) * times[itime])
                self.graphs.append(g)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    
#%%

fig1 = plt.figure()
ax1 = plt.axes() 
for t in diffp:
    ax1.plot(diffp[t],c = 'black', alpha = 0.1)

ax1.set_ylim([-3,3])
plt.show()
#%%

dataset = AortaDataset(nodes, edges, lengths, gpressures, gvelocities, areas, times, diffp, diffq, edge_features, node_degree, r_gpressures, r_gvelocities, r_diffp, r_diffq)

num_examples = len(dataset)
num_train = int(num_examples * 1)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=1, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False)

from torch.nn.modules.module import Module
from dgl.nn import ChebConv
from dgl.nn import GraphConv
from dgl.nn import GATConv
from dgl.nn import RelGraphConv
from torch.nn import LayerNorm
from torch.nn import Linear
import torch.nn.functional as F
import dgllife
import dgl.function as fn
import torch.optim as optim

class GCN(Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        # lm = dgl.laplacian_lambda_max(g)
        # self.conv1 = ChebConv(in_feats, h_feats, 3, bias = False)
        # self.conv1 = GATConv(h_feats, h_feats, num_heads = 1, feat_drop = 0.1)
        self.num_heads = 2
        self.conv1 = GATConv(in_feats, h_feats, num_heads = self.num_heads, feat_drop = 0.5)
        self.conv2 = GATConv(h_feats, 2, num_heads = 1, feat_drop = 0.5)
        # self.conv3 = GraphConv(h_feats, 2)
        self.double()
        self.lms = {}

    def forward(self, g, in_feat):
        if g.num_nodes() not in self.lms:
            self.lms[g.num_nodes()] = dgl.laplacian_lambda_max(g)
        # h = self.conv1(g, in_feat, lambda_max = self.lms[g.num_nodes()])
        h = self.conv1(g, in_feat)
        h = torch.sum(h, dim = 1)
        h = h / self.num_heads
        h = F.relu(h)
        h = self.conv2(g, h)
        # h = F.relu(h)
        # h = self.conv3(g, h)
        # g.ndata['h'] = h
        return h

class MLP(Module):
    def __init__(self, in_feats, latent, n_h_layers, normalize = True):
        super().__init__()
        self.encoder_in = Linear(in_feats, latent).double()
        self.encoder_out = Linear(latent, latent).double()
        
        self.n_h_layers = n_h_layers
        self.hidden_layers = []
        for i in range(n_h_layers):
            self.hidden_layers.append(Linear(latent, latent).double())
        
        self.normalize = normalize
        if self.normalize:
            self.norm = LayerNorm(latent).double()
            
    def forward(self, inp):
        enc_features = self.encoder_in(inp)
        enc_features = F.relu(enc_features)
        
        for i in range(self.n_h_layers):
            enc_features = self.hidden_layers[i](enc_features)
            enc_features = F.relu(enc_features)
            
        enc_features = self.encoder_out(enc_features)
        
        if self.normalize:
            enc_features = self.norm(enc_features)
        
        return enc_features
        

class GraphNet(Module):
    def __init__(self, in_feats_nodes, in_feats_edges, latent, h_feats, L, hidden_layers):
        super(GraphNet, self).__init__()
        
        normalize_inner = True
        
        self.encoder_nodes = MLP(in_feats_nodes, latent, hidden_layers, normalize_inner)
        self.encoder_edges = MLP(in_feats_edges, latent, hidden_layers, normalize_inner)

        self.processor_edges = []
        self.processor_nodes = []
        for i in range(L):      
            self.processor_edges.append(MLP(latent * 3, latent, hidden_layers, normalize_inner))
            self.processor_nodes.append(MLP(latent * 2, latent, hidden_layers, normalize_inner))

        self.L = L

        self.output = MLP(latent, h_feats, hidden_layers, False)
        self.inlet_node = inlet_node
        self.outlet_nodes = outlet_nodes
        
    def encode_nodes(self, nodes):
        f = nodes.data['features_c']
        enc_features = self.encoder_nodes(f)
        return {'proc_node': enc_features}
    
    def encode_edges(self, edges):
        f = edges.data['dist']
        enc_features = self.encoder_edges(f)
        return {'proc_edge': enc_features}
    
    def process_edges(self, edges, layer):
        f1 = edges.data['proc_edge']
        f2 = edges.src['proc_node']
        f3 = edges.dst['proc_node']
        proc_edge = self.processor_edges[layer](torch.cat((f1, f2, f3),dim=1))
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge' : proc_edge}
    
    def process_nodes(self, nodes, layer):
        f1 = nodes.data['proc_node']
        f2 = nodes.data['pe_sum']
        proc_node = self.processor_nodes[layer](torch.cat((f1, f2),dim=1))
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
        
        # g.ndata['h'][self.inlet_node,1] = 0
        # g.ndata['h'][self.outlet_nodes,0] = 0
        return g.ndata['h']
    
def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

# Create the model with given dimensions
latent = 2
infeat_nodes = 4
infeat_edges = 4
# model = GCN(infeat_nodes, latent)
# setting hidden_layers != 0 may make the network difficult to train
model = GraphNet(infeat_nodes, infeat_edges, latent, 2, 0, hidden_layers = 0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
nepochs = 1000
for epoch in range(nepochs):
    print('ep = ' + str(epoch))
    global_loss = 0
    count = 0
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['features'].double()).squeeze()
        # loss = F.mse_loss(pred, torch.reshape(labels, pred.shape))
        weight = torch.ones(pred.shape) 
        # weight[inlet_node,0] = 10
        # weight[outlet_nodes,1] = 10
        # weight[inlet_node,1] = 0
        # weight[outlet_nodes,0] = 0
        # weight[:,1] = 0
        loss = weighted_mse_loss(pred, torch.reshape(labels, pred.shape), weight)
        global_loss = global_loss + loss.detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = count + 1
    if epoch % 1 == 0:
        fig1 = plt.figure()
        ax1 = plt.axes() 
        idx = 0
        ax1.plot(pred[:,idx].detach().numpy(),'-r')
        ax1.plot(torch.reshape(labels, pred.shape).detach().numpy()[:,idx],'-b')
        # ax1.set_ylim(np.min(torch.reshape(labels, pred.shape).detach().numpy()[:,idx]),
        #              np.max(torch.reshape(labels, pred.shape).detach().numpy()[:,idx]))
        ax1.set_title('loss p' + str(batched_graph.ndata['time'][0].numpy()))
        
        fig2 = plt.figure()
        ax2 = plt.axes()
        idx = 1
        ax2.plot(pred[:,idx].detach().numpy(),'-r')
        ax2.plot(torch.reshape(labels, pred.shape).detach().numpy()[:,idx],'-b')
        # ax2.set_ylim(np.min(torch.reshape(labels, pred.shape).detach().numpy()[:,idx]),
        #              np.max(torch.reshape(labels, pred.shape).detach().numpy()[:,idx]))
        ax2.set_title('loss q' + str(batched_graph.ndata['time'][0].numpy()))
        plt.show()
    scheduler.step()
            
    print('\tloss = ' + str(global_loss / count))
        
#%% when we predict delta 

it = iter(train_dataloader)
batch = next(it)
batched_graph, labels = batch
graph = dgl.unbatch(batched_graph)[0]
timeg = float(graph.ndata['time'][0])
nodes_degree = np.expand_dims(graph.ndata['features'][:,-1],1)
    
tin = 240
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

    if _normalize:
        # prev = torch.from_numpy(np.hstack((tp, tq, gp, gq, areas, nodes_degree)))
        pred = model(graph, prev)
        # pred = find_solutions(model, graph, prev, prev, outlet_nodes, inlet_node, bcs_p, bcs_q)
        pred = pred.squeeze()
        deltap = mp + pred[:,0].detach().numpy() * (Mp - mp)
        predp = deltap + mp + prev[:,0].numpy() * (Mp - mp)
        truep = mp + gpressures[nt] * (Mp - mp)
        fig1 = plt.figure()
        ax1 = plt.axes() 
        ax1.plot(predp / 1333.2,'r')
        ax1.plot(truep / 1333.2,'-b')
        ax1.set_title('pressure t = ' + str(t))
        # ax1.set_ylim((mp / 1333.2,Mp / 1333.2))
        ax1.set_xlim((0, nodes.shape[0]))
        plt.savefig('pressure/t' + str(count).rjust(3, '0'))
    
        deltaq = mq + pred[:,1].detach().numpy() * (Mq - mq)
        predq = deltaq + mq + prev[:,1].numpy() * (Mq - mq)
        trueq = mq + gvelocities[nt] * (Mq - mq)
        fig2 = plt.figure()
        ax2 = plt.axes()
        ax2.plot(predq,'r')
        ax2.plot(trueq,'--b')
        ax2.set_title('velocity t = ' + str(t))
        # ax2.set_ylim((mq,Mq))
        ax2.set_xlim((0, nodes.shape[0]))
        plt.savefig('flowrate/t' + str(count).rjust(3, '0'))
        plt.show()
    else:
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
        ax2.set_title('flowrate t = ' + str(t))
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