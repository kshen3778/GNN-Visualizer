import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from plotly import graph_objs as go
from sklearn.manifold import TSNE

def runAndExtract(model, layerNums, *model_inputs):

    hidden_vectors = []
    def extractHidden(self, input, output):
        hidden_vectors.append(output.data.detach().numpy())

    handles = []
    all_layer_ids = list(dict(model.named_children()).keys())
    for i, layer in enumerate(model.children()):
        if(i in layerNums):
            #layerId = all_layer_ids[i]
            handles.append(layer.register_forward_hook(extractHidden))

    #run model
    output = model(*model_inputs)


    #remove handles
    for h in handles:
        h.remove()


    return output, hidden_vectors

def tsne_reduce(data, params):
    tokens = np.array(data)
    tsne_model = TSNE(perplexity=params["perplexity"], n_components=params["n_components"], init=params["init"], n_iter=params["n_iter"], random_state=params["random_state"])
    new_values = tsne_model.fit_transform(tokens)
    return new_values

def generateGraph(edges, nodes, embeddings, scores = None, labelled_nodes = None, tsne_params = {"perplexity": 25, "n_components": 2, "init": "pca", "n_iter": 2500, "random_state": 25}):
    G=nx.Graph()
    print("==== Performing Dimensionality Reduction ===")
    tsne_pos = tsne_reduce(embeddings, tsne_params)
    tsne_pos_dict = {}
    labelled_idx_names = []
    if(labelled_nodes):
        labelled_idx_names = [nodes[x] for x in labelled_nodes]
    #edges = edges.to_dense().numpy()
    # =============init nodes
    x = 0
    y = 0
    nd_cnt = 0
    print("Initializing Nodes")
    for i in range(len(nodes)): # loop through all the nodes
        #print(nd_cnt/len(scores))
        nd_cnt += 1
        nd_name = "NA" #idx_map2.get(i,"NA")

        if (i in nodes):
            nd_name = nodes[i]
        else:
            print("WARNING: couldn't find node with index: " + str(i))
        if(scores):
            G.add_node(
                nd_name,
                pos=(x,y),
                label=nd_name,
                score=scores[i])
        else:
            G.add_node(
                nd_name,
                pos=(x,y),
                label=nd_name)
        tsne_pos_dict[nd_name] = tsne_pos[i]
        if x == y:
            y += 1
            x = 0
        else:
            x += 1
    # ===========init edges
    edge_cnt = 0
    print("Initializing Edges")
    for edge in edges:
        #print(edge_cnt/len(edges))
        edge_cnt += 1
        nd_name1 = nodes.get(int(edge[0]),"NA")
        nd_name2 = nodes.get(int(edge[1]),"NA")
        if(nd_name1 == "NA" or nd_name2 == "NA"):
            print("WARNING: edge has no name in index map: ", edge)
        G.add_edge(nd_name1,nd_name2,weight=0) # weight = 0 cause edges have no weight



    #labels = nx.get_edge_attributes(G,'weight')
    #pos=nx.get_node_attributes(G,'pos')
    #nx.draw(G,pos)



    length = nx.get_edge_attributes(G, 'length')
    #graph_layout = nx.spring_layout(G)
    #graph_layout = nx.random_layout(G)
    #graph_layout = nx.circular_layout(G)
    graph_layout = tsne_pos_dict


    # ==========plotting the graph

    #fig1 = figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

    #nx.draw(G, pos=nx.spring_layout(G))
    #nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels=length)
    #nx.draw_networkx_labels(G,pos=nx.spring_layout(G))

    print("==== Generating Figures ====")
    print("")
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        #mode='markers+text',
        mode='markers',
        hoverinfo='text',
        textposition='top center',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Hot',
            #color_continuous_scale=px.colors.sequential.Viridis,
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2, color=[])))


    for node in G.nodes():
        if(G.node[node] == {}):
            print("WARNING: couldn't find node ", node)
            continue
        x, y = graph_layout[G.node[node]['label']]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        if(scores):
            node_trace['text'] += tuple([str(G.nodes[node]["label"]) + "<br>" + str(G.nodes[node]["score"])])
            node_trace['marker']['color'] += tuple(G.nodes[node]["score"].tolist())
        else:
            node_trace['text'] += tuple([str(G.nodes[node]["label"])])

        if(node in labelled_idx_names):
            node_trace['marker']['line']['color'] += tuple(["LightGreen"])
        else:
            node_trace['marker']['line']['color'] += tuple(["Black"])

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        text=[],
        textposition='bottom center',
        #mode='lines+text'
        mode='lines'
    )

    for edge in G.edges():
        if(G.node[edge[0]] == {} or G.node[edge[1]] == {}):
            print("WARNING: couldn't find edge", edge)
            continue
        x0, y0 = graph_layout[G.node[edge[0]]['label']]
        x1, y1 = graph_layout[G.node[edge[1]]['label']]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_trace['text'] += tuple([G.get_edge_data(edge[0],edge[1])["weight"]])

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Graph',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig
