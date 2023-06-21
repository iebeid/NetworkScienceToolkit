
import itertools
import random
import multiprocessing
from collections import defaultdict
from collections import deque
import networkx as nx
from community import community_louvain as louvain
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import identity
from scipy.sparse.csgraph import connected_components
# from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


class RandomWalks():

    # Constructor
    def __init__(self, graph):
        self.graph = graph

    # Simple naive breadth first traversal of a graph
    def breadth_first_search(self, root):
        visited = set()
        visited.add(root)
        queue = deque([root])
        while queue:
            dequeued_node = queue.popleft()
            neighbors = self.graph.tasks.get_neighbors(dequeued_node)
            for neighbour in neighbors:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.extend(neighbors)
        return list(visited)



    def akef_unbiased_random_walker(self, walk_length):
        walks = []
        for node in tqdm(self.graph.nodes.keys()):
            neighbours = self.graph.tasks.get_neighbors(node)
            if neighbours:
                number_of_walks_per_node = len(neighbours)
                for _ in range(number_of_walks_per_node):
                    walk = [node]
                    current_node = node
                    for _ in range(walk_length - 1):
                        neighbours = self.graph.tasks.get_neighbors(current_node)
                        if len(neighbours) == 0:
                            # dead end, so stop
                            break
                        else:
                            # has neighbours, so pick one to walk to
                            current_node = random.choice(neighbours)
                        walk.append(current_node)
                    walks.append(walk)
                    if walk:
                        walks.append(walk)
            else:
                walks.append([node])
        return walks

    def akef_metapath_triples_biased_random_walker(self, metapaths, walk_length, walks_per_node):
        walks = []
        for metapath in tqdm(metapaths):
            minimum_lengths_of_metapath = len(metapath)
            metapath_type = ""
            if metapath[0] == metapath[-1]:
                metapath_type = "s"
            else:
                metapath_type = "a"
            if metapath_type == "s":
                metapath = [metapath[0]] + list(metapath[1:] * ((walk_length // (len(metapath) - 1)) + 1))
            if metapath_type == "a":
                metapath = metapath * ((walk_length // (len(metapath) - 1)) + 1)
            for _ in range(walks_per_node):
                for node in self.graph.nodes.keys():
                    current_node = node
                    node_type = self.graph.nodes[node]["type"]
                    walk = ([])
                    if node_type == metapath[0]:
                        for d in range(len(metapath) - 1):
                            walk.append(current_node)
                            neighbours = self.graph.tasks.get_neighbors(current_node)
                            filtered_neighbors = []
                            for n_node in neighbours:
                                n_node_type = self.graph.nodes[n_node]["type"]
                                if n_node_type == metapath[d + 1]:
                                    filtered_neighbors.append(n_node)
                            if len(filtered_neighbors) == 0:
                                break
                            current_node = random.choice(filtered_neighbors)
                    if len(walk) >= minimum_lengths_of_metapath:
                        walks.append(walk)
        return walks

    def akef_weighted_biased_random_walker(self, walk_length):
        walks = []
        for node in tqdm(self.graph.nodes.keys()):
            neighbours = self.graph.tasks.get_neighbors(node)
            if len(neighbours) > 0:
                number_of_walks_per_node = len(neighbours)
                for _ in range(number_of_walks_per_node):
                    walk = [node]
                    current_node = node
                    for _ in range(walk_length - 1):
                        neighbours = self.graph.tasks.get_neighbors(current_node)
                        neighbours_weights = self.graph.tasks.get_neighborhood_weights(current_node)
                        s = self.graph.tasks.get_sum_of_neighborhood_weights(current_node)
                        p_v = []
                        for n_w in neighbours_weights:
                            p_v.append(n_w / s)
                        if len(neighbours) == len(p_v):
                            current_node = np.random.choice(neighbours, p=p_v)
                        walk.append(current_node)
                    walks.append(walk)
                    if walk:
                        walks.append(walk)
            else:
                walks.append([node])
        return walks


    def spectral_random_walker(KG, walk_length, number_of_walks):
        size_of_nodes = len(list(KG.nodes()))
        A = nx.adjacency_matrix(KG)
        A = A.todense()
        A = np.array(A, dtype=np.float64)
        D = np.diag(np.sum(A, axis=0))
        T = np.dot(np.linalg.inv(D), A)
        walks = []
        for _ in tqdm(range(number_of_walks)):
            p_state = np.zeros((size_of_nodes,), dtype=int).reshape(-1, 1)
            random_index = np.random.choice(np.arange(0, size_of_nodes), 1, replace=True)
            p_state[random_index] = 1
            visited = []
            for _ in range(walk_length):
                p_state = np.dot(T, p_state)
                visited.append(np.argmax(p_state))
            walks.append(visited)
        return walks

    def spectral_parallelized_random_walker(KG, walk_length, number_of_walks):
        size_of_nodes = len(list(KG.nodes()))
        A = nx.adjacency_matrix(KG)
        A = A.todense()
        A = np.array(A, dtype=np.float64)
        D = np.diag(np.sum(A, axis=0))
        T = np.dot(np.linalg.inv(D), A)
        walks = []
        p_state = np.eye(size_of_nodes, number_of_walks, dtype=int).reshape(-1, 1)
        # for _ in tqdm(range(number_of_walks)):
        # random_index = np.random.choice(np.arange(0,size_of_nodes),1,replace=False)
        # p_state[random_index]=1
        visited = []
        for _ in range(walk_length):
            p_state = np.dot(T, p_state)
            visited.append(np.argmax(p_state))
        walks.append(visited)
        return walks

    # markov chain based random walker
    def markovian_random_walker(self, walk_length, number_of_walks, restart_probability):
        inv_D = scipy.sparse.linalg.inv(self.graph.degree)
        T = inv_D.dot(self.graph.adjacency)
        p_state = scipy.sparse.lil_matrix((self.graph.N, number_of_walks), dtype=np.int)
        for m in range(number_of_walks):
            random_index = np.random.choice(np.arange(0, self.graph.N), 1, replace=True)
            p_state[random_index, m] = 1
        p_state = p_state.tocsc()
        v_initial = p_state
        walks_matrix = []
        for _ in tqdm(range(walk_length)):
            p_state = ((1 - restart_probability) * T.dot(p_state)) + (restart_probability * v_initial)
            p_state_max = p_state.argmax(axis=0)
            p_state_max = p_state_max.tolist()[0]
            walks_matrix.append(p_state_max)
        return np.array(walks_matrix).transpose().tolist()

    # heterogenous markov chain based random walker
    def heterogenous_markovian_random_walker(self, walk_length, number_of_walks, restart_probability):
        inv_D = scipy.sparse.linalg.inv(self.graph.degree)
        T = inv_D.dot(self.graph.A_schema)
        p_state = scipy.sparse.lil_matrix((self.graph.N, number_of_walks), dtype=np.int)
        for m in range(number_of_walks):
            random_index = np.random.choice(np.arange(0, self.graph.N), 1, replace=True)
            p_state[random_index, m] = 1
        p_state = p_state.tocsc()
        v_initial = p_state
        walks_matrix = []
        for _ in tqdm(range(walk_length)):
            p_state = ((1 - restart_probability) * T.dot(p_state)) + (restart_probability * v_initial)
            p_state_max = p_state.argmax(axis=0)
            p_state_max = p_state_max.tolist()[0]
            walks_matrix.append(p_state_max)
        return np.array(walks_matrix).transpose().tolist()


class GATNERandomWalker():
    def __init__(self, nx_G, node_type_arr=None, num_workers=16):
        self.G = nx_G
        self.node_type = node_type_arr
        self.num_workers = num_workers

    def walk(self, args):
        walk_length, start, schema = args
        # Simulate a random walk starting from start node.
        rand = random.Random()

        if schema:
            schema_items = schema.split('-')
            assert schema_items[0] == schema_items[-1]

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur]:
                if schema == '' or node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return [str(node) for node in walk]

    def initializer(self, init_G, init_node_type):
        global G
        G = init_G
        global node_type
        node_type = init_node_type

    def node_list(self, nodes, num_walks):
        for loop in range(num_walks):
            for node in nodes:
                yield node

    def simulate_walks(self, num_walks, walk_length, schema=None):
        all_walks = []
        nodes = list(self.G.nodes())
        random.shuffle(nodes)

        if schema is None:
            with multiprocessing.Pool(self.num_workers, initializer=self.initializer,
                                      initargs=(self.G, self.node_type)) as pool:
                all_walks = list(
                    pool.imap(self.walk,
                              ((walk_length, node, '') for node in tqdm(self.node_list(nodes, num_walks))),
                              chunksize=256))
        else:
            schema_list = schema.split(',')
            for schema_iter in schema_list:
                with multiprocessing.Pool(self.num_workers, initializer=self.initializer,
                                          initargs=(self.G, self.node_type)) as pool:
                    walks = list(pool.imap(self.walk, ((walk_length, node, schema_iter) for node in
                                                       tqdm(self.node_list(nodes, num_walks)) if
                                                       schema_iter.split('-')[0] == self.node_type[node]),
                                           chunksize=512))
                all_walks.extend(walks)

        return all_walks