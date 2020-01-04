# GNN-Visualizer
A toolkit for extracting and visualizing the hidden layer feature vectors of graph neural networks.

# How to use

``` runAndExtract(model, layerNums, *model_inputs): ```

Will do forward propagation as well as return the feature vectors right after the forward call for all layers in layerNums (in the order that you specify them in the array). This function essentially replaces the normal forward call such as out = model(input). 

model: The Pytorch model.

layerNums: The layers you want to extract the feature vectors from (the layer id corresponds to the declaration order in the module, see example below).

*model_inputs: Can take in an arbitrary amount of inputs for your model depending on your model.

Example:
```
#GCN model from https://github.com/tkipf/pygcn
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
      
        self.gc1 = GraphConvolution(nfeat, nhid) #this layer (gc1) is id 0
        self.gc2 = GraphConvolution(nhid, nclass) #this layer (gc2) is id 1
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
        
model = GCN(...)
optimizer = optim.Adam(...)
        
#Inside your training file:

from visualize import runAndExtract

t = time.time()
model.train()
optimizer.zero_grad()
#The runAndExtract method will extract the feature vectors after gc1 and gc2's forward calls
output, h_feat = runAndExtract(model, [0, 1], features, adj) #This line replaces output = model(features, adj)
#h_feat will contain the hidden feature vectors in the order of layerNums (in this case for layer id 0 and then layer id 1)
# ... etc ...   
```

----------------------------------------


```
generateGraph(edges, nodes, embeddings, scores = None, labelled_nodes = None, tsne_params = {"perplexity": 25, "n_components": 2, "init": "pca", "n_iter": 2500, "random_state": 25}):
```
Uses Plotly to generate an interactive visualization.

edges: An array of edges in the form of:
[[node1 node2], #an edge connects node1 to node2
[node2 node3],
..etc..
]

nodes: A map of id to nodes:
{ 0: "node1",
1: "node2", 
etc...
}

embeddings: A Numpy array of the vector embedding for each node in the nodes mapping (must match the same order as the mapping).

scores: Optional prediction/class value for each node (useful for displaying final predictions or classifications during test).

labelled_nodes: Optional, if you want to mark certain nodes on the graph.

tsne_params: Our visualization uses sklearn.manifold.TSNE for dimensionality reduction. If you don't want the default, you can pass in your own parameters as a mapping as shown above.





