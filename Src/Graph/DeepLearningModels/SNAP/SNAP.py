import snap

import random
import numpy as np
import matplotlib.pyplot as plt

# Graph = snap.GenRndGnm(snap.PNGraph, 10, 10)

# Graph = snap.TNGraph.New()
# Graph.AddNode(1)
# Graph.AddNode(5)
# Graph.AddNode(32)
# Graph.AddEdge(1,5)
# Graph.AddEdge(5,32)
# Graph.AddEdge(32,1)

# # create a random walk between states
# prob = [0.05, 0.95]
# # statically defining the starting position
# start = 2
# positions = [start]
# # creating the random points
# rr = np.random.random(1000)
# downp = rr < prob[0]
# upp = rr > prob[1]
# # creating the random points
# rr = np.random.random(1000)
# downp = rr < prob[0]
# upp = rr > prob[1]
# for idownp, iupp in zip(downp, upp):
#     down = idownp and positions[-1] > 1
#     up = iupp and positions[-1] < 4
#     positions.append(positions[-1] - down + up)
# # plotting down the graph of the random walk in 1D
# plt.plot(positions)
# plt.show()

# create a directed random graph on 100 nodes and 1k edges
G2 = snap.GenRndGnm(snap.PNGraph, 100, 200)
# traverse the nodes
for NI in G2.Nodes():
    print("node id %d with out-degree %d and in-degree %d" % (NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))
# traverse the edges
for EI in G2.Edges():
    print("edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
# traverse the edges by nodes
for NI in G2.Nodes():
    for Id in NI.GetOutEdges():
        print("edge (%d %d)" % (NI.GetId(), Id))
snap.DrawGViz(G2, snap.gvlDot, "graph2.png", "graph 1")

#-------------------------------------------------------------