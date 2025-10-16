import numpy as np
import pandas as pd
import time
import progressbar
import networkx as nx
from networkx.algorithms.link_analysis import pagerank
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.centrality import degree_centrality, in_degree_centrality, out_degree_centrality, betweenness_centrality

class Metrics:
    def __init__(self):
        pass

    def degree_centrality(self, graph):
        ''' Degree Centrality metric for graph analysis '''
        # centralities = {}
        # for node in graph.nodes:
        #     in_degree = graph.in_degree(node.link)
        #     out_degree = graph.out_degree(node.link)
        #     n_edges =  in_degree + out_degree
        #     centralities[node.link] = [n_edges, out_degree, in_degree, n_edges/len(graph.edges)]

        all_deg = degree_centrality(graph)
        in_deg = in_degree_centrality(graph)
        out_deg = out_degree_centrality(graph)
        result = [[key, all_deg[key], out_deg[key], in_deg[key]] for key in all_deg]
        result = sorted(result, key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(result, columns=['Node', 'All', 'Outgoing','Incoming'])
        df.to_csv('results/centralities.csv', index=False)

        return result

    def pagerank(self, graph, n_iter=10, save=True):
        ''' Pagerank algorithm '''
        # idx_to_link = {}
        # link_to_idx = {}
        # for i, key in enumerate(graph.nodes.keys()):
        #     idx_to_link[i] = key
        #     link_to_idx[key] = i


        # nodes = list(graph.nodes.values())
        # N = len(nodes)

        # rankings = np.ones((N, 1))
        # rankings /= N

        # # need to do matrix mult in batches since full matrix is (~300,000, ~300,000) -> Too big for memory
        # # if can load in memory, matrix mult of whole would have been easier
        # batch_size = 100

        # # perform n iterations of pagerank
        # for _ in range(n_iter):
        #     new_rankings = np.zeros((N, 1))

        #     for i in range(0, N, batch_size):
        #         batch_nodes = nodes[i:i+batch_size]
        #         batch_matrix = np.zeros((len(batch_nodes), N))
        #         for _, node in enumerate(batch_nodes):
        #             batch_matrix = self._calculate_batch_matrix(batch_matrix, node, link_to_idx, i)

        #         new_rankings += np.dot(np.transpose(batch_matrix), rankings[i:i+batch_size])
        #     rankings = new_rankings

        # # match up the node to the pagerank
        # result = {}
        # rankings = rankings.reshape(rankings.shape[0],)
        # for idx, ranking in enumerate(rankings):
        #     link = idx_to_link[idx]
        #     result[link] = rankings[idx]

        result = pagerank(graph)
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        if save:
            df = pd.DataFrame(result, columns=['Node', 'Score'])
            df.to_csv('results/pageranks.csv', index=False)

        return result


    def rank_neighbors(self, graph, source, edge_type='bind'):

        pageranks = self.pagerank(graph, save=False)
        pageranks = dict(pageranks)

        neighbor_scores = []
        for neighbor in graph.neighbors(source):
            edge = graph.edges[source, neighbor]
            if edge['edge_type'] == edge_type:
                score = pageranks[neighbor]
                neighbor_scores.append((neighbor, score))

        return sorted(neighbor_scores,  key=lambda x: x[1], reverse=True)


    # runtime: O(n*m + (n^2)*logn) where n = |V| = ~300,000 vertices and m = |E| = ~728,000 edges
    # runtime is too big to run this metric on the size of this graph
    def betweenness_centrality(self, graph):
        ''' Compute Betweenness Centralitities for each node in graph '''
        # betweenness = dict.fromkeys(graph.nodes, 0.)

        # bar = progressbar.ProgressBar(max_value=len(graph.nodes))
        # for i, s in enumerate(graph.nodes.keys()):
        #     bar.update(i)
        #     pred = dict.fromkeys(graph.nodes, [])
        #     delta_v = dict.fromkeys(graph.nodes, 0)
        #     dist = dict.fromkeys(graph.nodes, -1)
        #     n_paths = dict.fromkeys(graph.nodes, 0)
        #     queue = []
        #     stack = []
        #     node = graph.nodes[s]
        #     queue.append(node)
        #     dist[s] = 0
        #     n_paths[s] = 1
        #     while queue:
        #         v = queue.pop(0)
        #         stack.append(v.link)
        #         neighbors = [w.dest for w in v.outgoing_edges]
        #         for w in neighbors:
        #             if dist[w.link] < 0:
        #                 dist[w.link] = dist[v.link] + 1
        #                 queue.append(w)
        #             if dist[w.link] == (dist[v.link] + 1):
        #                 n_paths[w.link] += n_paths[v.link]
        #                 pred[w.link].append(v.link)
        #     while stack:
        #         w = stack.pop()
        #         for v in pred[w]:
        #             delta_v[v] = delta_v[v] + (n_paths[v]/n_paths[w])*(1 + delta_v[w])
        #         if w != s:
        #             betweenness[w] += delta_v[w]

        betweenness = betweenness_centrality(graph)
        result = sorted(betweenness.items(), key=lambda x: x[1], reserve=True)
        df = pd.DataFrame(result, columns=['Node', 'Score'])
        df.to_csv('results/betweenness.csv', index=False)

        return result

    def shortest_path(self, graph, src, target):

        # dist, path, edges = self._BFS(graph, src, dest)
        # if dist == -1:
        #     print(f'No path between given source {src} and destination {dest}')
        graph = nx.Graph(graph)
        path = shortest_path(graph, src, target)
        return path

    # @staticmethod
    # def _calculate_batch_matrix(batch_matrix, node, link_to_idx, curr_idx):
    #     ''' Helper function for pagerank '''

    #     link = node.link

    #     # Guarantee idx of node matches up with node link
    #     idx = link_to_idx[link]
    #     idx -= curr_idx
    #     edges = node.outgoing_edges
    #     if len(edges) != 0:
    #         for edge in edges:
    #             dest_link = edge.dest.link
    #             dest_idx = link_to_idx[dest_link]
    #             batch_matrix[idx, dest_idx] = 1
    #         batch_matrix[idx] /= len(edges)

    #     return batch_matrix

    # @staticmethod
    # def _BFS(graph, src, dest):
    #     visited = dict.fromkeys(graph.nodes, False)
    #     pred = dict.fromkeys(graph.nodes, [-1, -1])
    #     dist = dict.fromkeys(graph.nodes, np.inf)
    #     queue = []

    #     visited[src] = True
    #     dist[src] = 0
    #     queue.append(graph.nodes[src])

    #     while queue:
    #         v = queue.pop(0)
    #         # print(f'Node: {v.link}')
    #         outgoing_edges = [w.dest for w in v.outgoing_edges]
    #         incoming_edges = [w.origin for w in v.incoming_edges]
    #         neighbors = outgoing_edges + incoming_edges
    #         # print(f'neighbors {[n.link for n in neighbors]}\n')
    #         # print(len(neighbors))
    #         for n in neighbors:
    #             if visited[n.link] == False:
    #                 visited[n.link] = True
    #                 dist[n.link] = dist[v.link] + 1
    #                 for edge in (v.outgoing_edges + v.incoming_edges):
    #                     cond1 = edge.origin.link == v.link and edge.dest.link == n.link
    #                     cond2 = edge.origin.link == n.link and edge.dest.link == v.link
    #                     if cond1 or cond2:
    #                         e = edge

    #                 pred[n.link] = [v.link, e]
    #                 queue.append(n)

    #                 if(n.link == dest):
    #                     path = []
    #                     edges = []
    #                     u = n.link
    #                     path.insert(0, u)
    #                     while src not in path:
    #                         [u, e] = pred[u]
    #                         if u == -1:
    #                             break
    #                         path.insert(0, u)
    #                         edges.insert(0, e)

    #                     return dist[n.link], path, edges

    #     return -1, [], []
