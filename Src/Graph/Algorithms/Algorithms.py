from Src.Graph.Algorithms.CommunityDetection import CommunityDetection
from Src.Graph.Algorithms.ConnectedComponents import ConnectedComponents
from Src.Graph.Algorithms.Factorization import Factorization
from Src.Graph.Algorithms.RandomWalks import RandomWalks

class Algorithms():

    # Constructor
    def __init__(self, graph):
        self.graph = graph
        self.connected_components=ConnectedComponents(self.graph)
        self.community_detection=CommunityDetection(self.graph)
        self.factorization=Factorization(self.graph)
        self.random_walks=RandomWalks(self.graph)