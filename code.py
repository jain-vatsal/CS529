# -*- coding: utf-8 -*-
"""Another copy of GUSD.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nyyGUOyx0XdMabA-E5OnqQSE4sB-h7Ti

## Imports
"""

!pip install torch torchvision torchaudio
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
!pip install torch-geometric

# Commented out IPython magic to ensure Python compatibility.
import torch

import warnings
warnings.filterwarnings("ignore")

import torch_geometric
torch_geometric.__version__

import random

from torch_geometric.data import Data
from numpy.lib.index_tricks import index_exp
import numpy as np
import pandas as pd
from torch._C import dtype
import torch.nn.functional as F
import copy
import regex as re
import networkx as nx

from sklearn.cluster import KMeans, MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, adjusted_rand_score
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_edge_index, remove_self_loops

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold,cross_val_predict,cross_val_score,GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from torch_geometric.nn import GCNConv,GATConv
#import torch_sparse
from torch import FloatTensor

from re import A
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from scipy import sparse as sp

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

"""## Code

### Dataset class with pygeo graph
"""

class dataset_vibgnn():
  def __init__(self,edge_list,edge_weights,n_nodes,classes=None,features = None,verbose = True):

    self.n_nodes = n_nodes
    self.edge_list = edge_list

    if verbose :
      print("Building the graph...")

    if features is None:
      x = torch.tensor(np.identity(self.n_nodes), dtype=torch.float) #maybe in built funciton for py
    else:
      x = torch.tensor(features, dtype=torch.float)

    unzipped_el = list(zip(*edge_list))
    edge_index = torch.tensor([unzipped_el[0],unzipped_el[1]], dtype=torch.long)

    # print('---> Graph')

    if classes is not None:
      self.graph = Data(x=x,edge_index=edge_index,edge_weights = edge_weights,y = torch.tensor(classes))

    else:
      self.graph = Data(x=x,edge_index=edge_index)


  def build_negatives(self,n=10):
    #Dirty : could be optimized

    unzipped_el = list(zip(*self.edge_list))
    unzipped_el[0] = np.array(unzipped_el[0])
    unzipped_el[1] = np.array(unzipped_el[1])

    self.pos_examples = []
    self.neg_examples = []

    nodes_id = np.arange(0,self.n_nodes)



    for i in self.edge_list:
      self.pos_examples += [(i[0],i[1],1)]

      list_nei = unzipped_el[1][unzipped_el[0] == i[0]]
      arr = np.delete(nodes_id, list_nei)
      indice_negative = np.random.choice(arr, n, replace=False)

      self.neg_examples += [(i[0],j,0) for j in indice_negative]
  def build_negatives_hetero(self,edge_neg,n = 10):
    #Dirty : could be optimized

    unzipped_el = list(zip(*self.edge_list))
    unzipped_el[0] = np.array(unzipped_el[0])
    unzipped_el[1] = np.array(unzipped_el[1])

    self.pos_examples = []
    self.neg_examples = []

    nodes_id = np.arange(0,self.n_nodes)

    self.neg_examples = [(i[0],i[1],0) for i in edge_neg]
    for bb in range(n):
      self.neg_examples += [(i[0],i[1],0) for i in edge_neg]



    for i in self.edge_list:
      self.pos_examples += [(i[0],i[1],1)]

  def build_train(self,edge_tohide=None):
    if edge_tohide is not None:
      graph_train = copy.deepcopy(self.graph)
      edge_list_sub = list(set(self.edge_list)^set(edge_tohide))
      unzipped_el = list(zip(*edge_list_sub))
      edge_index = torch.tensor([unzipped_el[0],unzipped_el[1]], dtype=torch.long)
      graph_train.edge_index = edge_index
      return graph_train
    else:
      return self.graph

def build_adjacency_matrix(edges, nb_nodes=None):
    if nb_nodes is None:
        nb_nodes = np.max(edges) + 1
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.ones(rows.shape[0], dtype=np.int)
    A = sp.csr_matrix((data, (rows, cols)), shape=(nb_nodes, nb_nodes))

    assert(A.data.max() == 1)
    return A

"""### Various Functions"""

def conf_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR = tp/(tp+fn)
    return TPR

def eval(model,dataset,x_1_test,x_2_test,y_test):

  with torch.no_grad():
    model.eval()

    proba_p = model(dataset.graph.x, dataset.graph.edge_index,x_1_test,x_2_test)
    y_pred = torch.squeeze(proba_p).numpy()
    y_pred_argmax = np.round(y_pred).astype(int)

    auc_link=roc_auc_score(y_test, y_pred )

  return auc_link

"""### Models"""

class GAT(torch.nn.Module):
    def __init__(self, num_features,hidden_channels,heads,c3 = False):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels[0], heads = heads)  # TODO
        self.conv2 = GATConv(hidden_channels[0], hidden_channels[1],heads = heads)  # TODO
        self.conv3 = GATConv(hidden_channels[1], hidden_channels[1],heads = heads)  # TODO
        self.c3 = c3
        self.bn = torch.nn.BatchNorm1d(num_features=hidden_channels[0])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)

        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        if self.c3:
          x = x.relu()
          x = F.dropout(x, p=0.2, training=self.training)
          x = self.conv3(x, edge_index)

        return x

class GCN(torch.nn.Module):
    def __init__(self, num_features,hidden_channels):
        super().__init__()

        self.conv1 = GCNConv(num_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class mlp(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(mlp, self).__init__()
            self.do = torch.nn.Dropout(p=0.2)
            self.fc1 = torch.nn.Linear(input_size, input_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(input_size, output_size)
            self.tanh = torch.nn.Tanh()
            self.fc3 = torch.nn.Linear(input_size, output_size)

        def forward(self, x):
            x = self.relu(self.fc1(self.do(x)))

            x = self.tanh(self.fc2(self.do(x)))

            x = self.fc3(self.do(x))
            return x

class VIB(torch.nn.Module):
    def __init__(self, num_features,hidden_channels,encoder = "GCN",c3 = False,n_nodes = 1,bibi = 0):
        super().__init__()
        if encoder == "GAT":
          self.gnn = GAT(num_features, hidden_channels,heads = 1,c3 = c3)
        else:
          self.gnn = GCN(num_features, hidden_channels, c3 = c3)
          if encoder != "GCN":
            print("unknown encoder, using GCN")

        self.log_a = torch.nn.Parameter(torch.Tensor([0]))

        self.b = torch.nn.Parameter(torch.rand(1))

        self.mlp_mu = mlp(hidden_channels[1], hidden_channels[1])
        self.mlp_logsigma = mlp(hidden_channels[1], hidden_channels[1])

        self.N = torch.distributions.Normal(0, 1)

        self.S = torch.nn.Parameter(torch.rand(n_nodes,2))

        self.M = torch.nn.Parameter(torch.rand(2,hidden_channels[1]))

        self.m = torch.nn.ReLU()

        self.bibi = bibi


    def distance(self,graph,edge_index,x_1,x_2):
      embedding = self.gnn(graph, edge_index)


      x_1 = torch.squeeze(x_1).long()
      x_2 = torch.squeeze(x_2).long()

      return torch.sum(torch.square(embedding[x_1] - embedding[x_2]),1)

    def forward(self, graph, edge_index,x_1,x_2):

        dist = self.distance(graph,edge_index,x_1,x_2)

        proba_p = torch.sigmoid( - torch.exp(self.log_a) * dist + self.b)

        return proba_p

    def loss(self, graph, edge_index,x_1,x_2,y,criterion):

        embedding = self.gnn(graph, edge_index)
        x_1 = torch.squeeze(x_1).long()
        x_2 = torch.squeeze(x_2).long()

        dist = torch.sum(torch.square(embedding[x_1] - embedding[x_2]),1)
        proba_p = torch.sigmoid( - torch.exp(self.log_a) * dist + self.b)

        loss_cl =  criterion(proba_p,torch.squeeze(y))
        loss_km = torch.mean(torch.square(embedding - F.softmax(self.S, dim=1) @ self.M))
        loss_reg = torch.mean(torch.square(self.M[0,:] - self.M[1,:]))

        loss = self.bibi * loss_km + loss_cl #- loss_reg

        return loss

    def transform_labels(self, labels): # probably a nicer way to do this, but let's roll with it
        new_labs = torch.Tensor(len(labels), 2)
        for i in range(len(labels)):
            if labels[i] == 0:
                new_labs[i,0] = 1
                new_labs[i,1] = 0
            else:
                new_labs[i,0] = 0
                new_labs[i,1] = 1
        return new_labs

def visualize(model,dataset,color=None):
  with torch.no_grad():
    model.eval()
    h = model.gnn(dataset.graph.x, dataset.graph.edge_index)
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    if color is None:
      color = dataset.graph.y
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

"""### Encode and Build Dataset"""

def batch_encode_ant(x, tokenizer,model, device, max_seq_len=128, batch_size=16, return_mask=False):
    if type(x) is not list:
        x = x.tolist()

    z = []

    for i in tqdm(range(0, len(x), batch_size)):

        bob = tokenizer(x[i: i + batch_size], return_tensors="pt", padding="max_length", max_length=max_seq_len,
                        truncation=True)
        x_i = bob["input_ids"]
        x_m = bob["attention_mask"]

        x_i = x_i.to(device)
        x_m = x_m.to(device)
        z.append(model(x_i,x_m ,output_hidden_states = True)['hidden_states'][-1][:,0,:].detach().cpu().numpy())
    z = np.vstack(z)
    return z

def preprocess(corpus):
  text = []
  for txt in corpus:
    txt = re.sub(r'(http\S+)', '', txt, re.I|re.A)
    txt = re.sub(r'(@\S+)', '', txt, re.I|re.A)
    txt = re.sub(r'&amp;', ', & ', txt, re.I|re.A)
    txt = re.sub(r'&gt;', '>', txt, re.I|re.A)
    txt = re.sub(r'\w(,)\w', ', ', txt, re.I|re.A)
    txt = re.sub(r'\\n', '', txt, re.I|re.A)
    txt = re.sub(r'\\\'', '\'', txt, re.I|re.A)
    text.append(txt)
  return text

def encode_and_build(txt, data, edge_list_ini, edge_list, edge_weights, inv, hom,use_text,X,lm):

    if use_text:
        if X is None:
          device = torch.device("cuda")

          tokenizer = AutoTokenizer.from_pretrained(lm)

          model = AutoModelForSequenceClassification.from_pretrained(lm)
          model.to(device)

          z = batch_encode_ant(txt, tokenizer, model, device, max_seq_len=256, return_mask=True)
          data["embedding"] = list(z)

          nodes = list(data['id'].unique())
          classes = []
          X=[]

          for i in nodes:
              dat = data.loc[data['id'] == i]
              X.append(np.mean(dat['embedding']))
              classes.append(round(np.mean(dat['label'])) - 1)

          X = np.vstack(X)
        nodes = list(data['id'].unique())
        classes = []
        for i in nodes:
          dat = data.loc[data['id'] == i]
          classes.append(round(np.mean(dat['label'])) - 1)
    else:
        nodes = list(data['id'].unique())
        classes = []
        for i in nodes:
            dat = data.loc[data['id'] == i]
            classes.append(round(np.mean(dat['label'])) - 1)
        X = None


    # Create an empty graph
    graph = nx.Graph()
    num_nodes = len(nodes)

    edge_list_neg = edge_list

    # Add nodes to the graph
    graph.add_nodes_from(range(num_nodes))
    # Add edges to the graph from the edge list
    for edge in edge_list_ini:
        source = int(edge[0])
        target = int(edge[1])
        weight = edge[2]
        graph.add_edge(source, target, weight=weight)


    self_loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(self_loops)

    if hom:
        if inv:
            A = nx.adjacency_matrix(graph)
            A += A @ A.T
            graph = nx.from_numpy_array(A)
            self_loops = list(nx.selfloop_edges(graph))
            graph.remove_edges_from(self_loops)

        df = nx.to_pandas_edgelist(graph)
        edge_list = list(zip(list(df['source'].values), list(df['target'].values)))
        edge_weights = list(df['weight'].values)
    else:
        A = nx.adjacency_matrix(graph)
        A.data = np.maximum(A.data, 0)
        A.data = np.clip(A.data, None, 1)

        B = A @ A.T
        B.data = np.maximum(B.data, 0)
        B.data = np.clip(B.data, None, 1)

        A = B - A
        A.data = np.maximum(A.data, 0)
        A.data = np.clip(A.data, None, 1)

        graph = nx.from_numpy_array(A)
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)

        df = nx.to_pandas_edgelist(graph)
        edge_list = list(zip(list(df['source'].values), list(df['target'].values)))
        edge_weights = list(df['weight'].values)

    if hom:
        dataset = dataset_vibgnn(edge_list, edge_weights, len(classes), classes, X, inv)
        dataset.build_negatives(10)
    else:
        dataset = dataset_vibgnn(edge_list, edge_weights, len(classes), classes, X, inv)
        dataset.build_negatives_hetero(edge_list_neg, 10)

    full = torch.vstack([torch.Tensor(dataset.neg_examples),torch.Tensor(dataset.pos_examples)])
    full = full[torch.randperm(full.shape[0])]

    nt = int(full.shape[0] * 0.95)
    train = full[0:nt]
    test = full[nt:]

    x_1_train,x_2_train,y_train = torch.split(train,1,dim = 1)
    y_train = torch.tensor(y_train, dtype=torch.float)
    x_1_test,x_2_test,y_test = torch.split(test,1,dim = 1)

    edges_tohide = np.hstack([x_1_test,x_2_test])

    edges_tohide = edges_tohide[torch.squeeze(y_test.int()).numpy() == 1].astype(int)
    edges_tohide =list(zip(edges_tohide[:,0],edges_tohide[:,1]))
    dataset_train = dataset.build_train(edges_tohide)
    if X is None:
        X = torch.tensor(np.identity(num_nodes), dtype=torch.float)

    return dataset, dataset_train, x_1_test, x_2_test, x_1_train, x_2_train, y_test, y_train, classes,X

"""### Run Model"""

def run_model(map_path, edge_path, inv, hom,use_txt,lr,X,lm,no_of_clusters):
    data = pd.read_csv(map_path,delimiter = "\t")
    edge_list_ini = np.genfromtxt(edge_path, usecols = [0,1,2])

    nodes = list(data['id'].unique())
    map = dict(zip(nodes,list(range(len(nodes)))))
    edge = []
    for i in edge_list_ini:
      try:
        edge.append((map[i[0]],map[i[1]],i[2]))
      except:
        bob = 0
    edge_list_ini = edge

    edge_weights = [i[2] for i in edge_list_ini]
    edge_list = [(int(i[0]),int(i[1])) for i in edge_list_ini]

    data = data.dropna()
    txt = list(data["rawTweet"])
    num_nodes = len(nodes)

    dataset, dataset_train, x_1_test, x_2_test, x_1_train, x_2_train, y_test, y_train, classes,X = encode_and_build(txt, data, edge_list_ini, edge_list, edge_weights, inv, hom,use_txt,X,lm)

    criterion = torch.nn.BCELoss()
    model = VIB(num_features = dataset.graph.num_features,hidden_channels=[100,50],encoder = "GAT",c3=False,n_nodes = num_nodes,bibi =0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    acc_p = eval(model,dataset,x_1_test,x_2_test,y_test)

    z = model.gnn(dataset.graph.x, dataset.graph.edge_index).detach().numpy()

    # kmeans = KMeans(n_clusters=no_of_clusters, random_state=0, n_init="auto").fit(z)
    # y_pred = kmeans.labels_

    # clustering = AgglomerativeClustering(n_clusters=no_of_clusters).fit(z)
    # y_pred = clustering.labels_

    clustering = GaussianMixture(n_components=no_of_clusters, random_state=0).fit(z)
    y_pred = clustering.predict(z)

    a1 = accuracy_score(classes, y_pred)
    a2 = accuracy_score(classes, 1 - y_pred)

    accc = max(a1,a2)
    acc_rec_final = []
    acc_final = []
    f_final = []
    in_final = []
    patience = 10
    ref_acc = 0
    z_save = None


    # kmeans = KMeans(n_clusters=no_of_clusters, random_state=0, n_init="auto").fit(z)
    # y_pred = kmeans.labels_

    # clustering = AgglomerativeClustering(n_clusters=no_of_clusters).fit(z)
    # y_pred = clustering.labels_

    clustering = GaussianMixture(n_components=no_of_clusters, random_state=0).fit(z)
    y_pred = clustering.predict(z)

    a1 = accuracy_score(classes, y_pred)
    a2 = accuracy_score(classes, 1 - y_pred)

    accc = max(a1,a2)

    # print("Accuracy text only :", round(accc, 4) * 100)

    if a1 > a2:
        best_pred = y_pred
    else:
        best_pred = 1 - y_pred
    _, _, f, _ = precision_recall_fscore_support(classes, best_pred, average='weighted')

    # print("F1 text only :", round(f, 4) * 100)


    for epoch in range(10000):
        model.train()

        proba_p = model(dataset.graph.x, dataset.graph.edge_index,x_1_test,x_2_test)

        optimizer.zero_grad()  # Clear gradients.

        #loss_vib : vib based loss, loss: soft cont loss without stochasticity

        loss = model.loss(dataset_train.x, dataset_train.edge_index,x_1_train,x_2_train,y_train,criterion)
        #loss,_,_ = model.loss_vib(dataset_train.x, dataset_train.edge_index,x_1_train,x_2_train,y_train,criterion)
        loss.backward()  # Derive gradients.
        optimizer.step()


        if epoch % 100 == 0:
            model.eval()
            acc_p = eval(model,dataset,x_1_test,x_2_test,y_test)
            z = model.gnn(dataset.graph.x, dataset.graph.edge_index).detach().numpy()

            # kmeans = KMeans(n_clusters=no_of_clusters, random_state=0, n_init="auto").fit(z)
            # clustering = AgglomerativeClustering(n_clusters=no_of_clusters).fit(z)
            clustering = GaussianMixture(n_components=no_of_clusters, random_state=0).fit(z)

            if model.bibi == 0:
                # y_pred = kmeans.labels_
                # y_pred = clustering.labels_
                y_pred = clustering.predict(z)

            else:
                y_pred = torch.argmax(model.S, dim=1).detach().numpy()

            a1 = accuracy_score(classes, y_pred)
            a2 = accuracy_score(classes, 1 - y_pred)

            accc = max(a1,a2)
            if a1 > a2:
                best_pred = y_pred
            else:
                best_pred = 1 - y_pred
            _, _, f, _ = precision_recall_fscore_support(dataset.graph.y, best_pred, average='weighted')

            acc_rec_final.append(acc_p)
            acc_final.append(accc)
            f_final.append(f)
            # in_final.append(kmeans.inertia_)
            #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, test AUC (Link prediction): {acc_p:.4f}, Accuracy: {accc:.4f}, F-macro: {f:.4f}, Inertie: {kmeans.inertia_:.4f}')

            if acc_p > ref_acc:
                patience = 10
                ref_acc = acc_p
                z_save = z
            else:
                patience -= 1
            if patience == 0:
                break

    return z_save,acc_rec_final,acc_final,f_final,in_final,X

"""## Run Model"""

# options : 'euro', 'timme', 'cd', 'conref'
dat = 'cd'
# topics for CD; options : 'all', 'abortion', 'marijuana', 'gayRights', or 'obama'
top = 'abortion'
# whether or not to use TIMME-All when running with TIMME; False runs TIMME-Pure
t_all = True
path = ''


if dat == 'cd':
    dat = dat + top

if dat == 'timme':
    if t_all:
        dat = dat + '_all'

map_path = path + dat + '_mapping.csv'
edge_path = path + dat + '_graph.txt'

X = None
# whether or not to use text embeddings
use_text = True
# whether or not to use the homophilic version; False runs the heterophilic version
hom = False
lr = 0.0001
# number of trials to run
max_t = 10
all_models = ["distilbert-base-uncased","microsoft/MiniLM-L12-H384-uncased","albert-base-v2","huawei-noah/TinyBERT_General_4L_312D","google/electra-small-discriminator"]
no_of_clusters = 2


for lm in all_models:

    best_acc = []
    best_f1 = []

    for i in range(max_t):
        emb, acc_rec_final, acc_final, f_final, in_final, X = run_model(map_path, edge_path, False, hom, use_text, lr, X, lm, no_of_clusters)
        # i = np.argmin(in_final[-10:])
        best_acc.append(acc_final[-10:][i])
        best_f1.append(f_final[-10:][i])

    if dat == 'timme':
        if not t_all:
            dat = dat + '_pure'

    print(f"\nRunning model with cluster size: {no_of_clusters}")
    print(f"Dataset: {dat}, Language Model: {lm}")
    if max_t > 1:
        print("Average accuracy:", round(np.mean(best_acc), 4) * 100, "\tStandard deviation:", round(np.sqrt(np.var(best_acc)), 4) * 100)
        print("Average weighted F1:", round(np.mean(best_f1), 4) * 100, "\tStandard deviation:", round(np.sqrt(np.var(best_f1)), 4) * 100)
    else:
        print("Accuracy:", round(best_acc[0]) * 100)
        print("Weighted F1:", round(best_f1[0]) * 100)

