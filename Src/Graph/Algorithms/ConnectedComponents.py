
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


class ConnectedComponents():

    # Constructor
    def __init__(self, graph):
        self.graph = graph

    # Kolb, L., Sehili, Z., & Rahm, E. (2014). Iterative computation of connected graph
    # components with MapReduce. Datenbank-Spektrum, 14(2), 107-117.
    def talburt_transitive_closure_connected_component_detection(self):
        pairList = self.graph.edges
        # Bootstap process by add reverse of all pairs to the pairList
        iterationCnt = 0
        clusterList = []
        for pair in pairList:
            clusterList.append((pair[0], pair[1]))
            pairRev = (pair[1], pair[0])
            clusterList.append(pairRev)
            pairSelf1 = (pair[0], pair[0])
            clusterList.append(pairSelf1)
            pairSelf2 = (pair[1], pair[1])
            clusterList.append(pairSelf2)
        # Change 2.10
        pairList = list(set(clusterList))
        # Sort pairs in order by the first position (the key)
        pairList.sort()
        # All of the pairs with same key are a Key Group
        clusterList = []
        moreWorkToDo = True
        iteration = 0
        while moreWorkToDo:
            moreWorkToDo = False
            iteration += 1
            # Add a caboose record to the end of the pairList
            caboose = ('---', '---')
            pairList.append(caboose)
            keyGroup = []
            for j in range(0, len(pairList) - 1):
                currentPair = pairList[j]
                keyGroup.append(currentPair)
                # Look ahead to the next key
                nextPair = pairList[j + 1]
                currentKey = currentPair[0]
                nextKey = nextPair[0]
                # When next key is different, at end of Key Group and ready to process keyGroup
                if currentKey != nextKey:
                    firstGroupPair = keyGroup[0]
                    firstGroupPairKey = firstGroupPair[0]
                    firstGroupPairValue = firstGroupPair[1]
                    # Add new pairs to clusterList from key groups starting with reversed pair and larger than 1 pair
                    keyGroupSize = len(keyGroup)
                    if firstGroupPairKey > firstGroupPairValue:
                        if keyGroupSize > 1:
                            moreWorkToDo = True
                            for k in range(keyGroupSize):
                                groupPair = keyGroup[k]
                                groupPairValue = groupPair[1]
                                newPair = (firstGroupPairValue, groupPairValue)
                                clusterList.append(newPair)
                                newReversePair = (groupPairValue, firstGroupPairValue)
                                clusterList.append(newReversePair)
                            # Decide if first pair of keyGroup should move over to clusterList
                            lastGroupPair = keyGroup[keyGroupSize - 1]
                            lastGroupPairValue = lastGroupPair[1]
                            if firstGroupPairKey < lastGroupPairValue:
                                clusterList.append(firstGroupPair)
                    else:
                        # pass other key groups forward to cluster list
                        clusterList.extend(keyGroup)
                    keyGroup = []
            pairList = []
            # Change 2.10
            pairList = list(set(clusterList))
            pairList.sort()
            iterationCnt += 1
            clusterList = []
        cluster_ids = {}
        for p in pairList:
            cluster_ids[p[1]] = p[0]
        node_label_profile = defaultdict(list)
        for key, val in sorted(cluster_ids.items()):
            node_label_profile[val].append(key)
        node_label_profile = dict(node_label_profile)
        components = []
        for v in node_label_profile.values():
            components.append(v)
        return components

    # D. J. Pearce, “An Improved Algorithm for Finding the Strongly Connected
    # Components of a Directed Graph”, Technical Report, 2005
    def scipy_depth_first_connected_component_detection(self):
        n_components, labels = connected_components(csgraph=self.graph.adjacency, directed=False,
                                                    return_labels=True)
        components = []
        membership = {}
        for i, l in enumerate(labels):
            membership[self.graph.node_inverted_index[i]] = l
        res = defaultdict(list)
        for key, val in sorted(membership.items()):
            res[val].append(key)
        for k, v in res.items():
            components.append(v)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return [cluster_list]

    # Nuutila, E., & Soisalon-Soininen, E. (1994). On finding the strongly connected components
    # in a directed graph. Information processing letters, 49(1), 9-14.
    def networkx_breadth_first_connected_component_detection(self):
        g = self.graph.to_networkx
        components = nx.connected_components(g)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    def akef_breadth_first_connected_component_detection(self):
        seen = set()
        components = []
        for n in self.graph.nodes.keys():
            if n not in seen:
                c = self.__breadth_first_search(n)
                seen.update(c)
                components.append(c)
        return components

    # QUICK CONNECTED COMPONENT DETECTION
    def akef_quick_connected_components_detection(self):
        components = []
        for ni_name, ni in self.graph.node_index.items():
            neighbor_indices_i = set(self.graph.adjacency[:, ni].indices)
            neighbor_indices_i.add(ni)
            intersections = []
            if len(neighbor_indices_i) > 1:
                for nj_name, nj in self.graph.node_index.items():
                    if ni != nj:
                        neighbor_indices_j = set(self.graph.adjacency[:, nj].indices)
                        neighbor_indices_j.add(nj)
                        if len(neighbor_indices_j) > 1:
                            if len(set.intersection(neighbor_indices_i, neighbor_indices_j)) != 0:
                                new_component = list(set.union(neighbor_indices_i, neighbor_indices_j))
                                intersections.append(new_component)
            component = list(set(list(itertools.chain.from_iterable(intersections))))
            component.sort()
            if len(component) > 0:
                components.append(component)
        res = []
        [res.append(x) for x in components if x not in res]
        cluster_list = []
        for cc in res:
            cc_resolved = []
            for n in cc:
                cc_resolved.append(self.graph.node_inverted_index[n])
            least_node = min(cc_resolved)
            for n in cc_resolved:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list


    # Fiedler, M. (1975). A property of eigenvectors of nonnegative symmetric matrices and
    # its application to graph theory. Czechoslovak mathematical journal, 25(4), 619-633.
    def akef_spectral_unsigned_fiedler_connected_component_detection(self):
        components = []
        eigenvals, eigenvecs = scipy.linalg.eigh(self.graph.laplacian_weighted_enriched.todense())
        eigenvecs = eigenvecs.real
        eigenvals = eigenvals.real
        fiedler_pos = np.where(eigenvals == np.sort(eigenvals)[1])[0][0]
        fiedler_vector = np.real(np.transpose(eigenvecs)[fiedler_pos])
        positive_values = []
        negative_values = []
        neutral_values = []
        for j, v in enumerate(fiedler_vector):
            if float(v) > float(0):
                positive_values.append(self.graph.node_inverted_index[j])
            if float(v) < float(0):
                negative_values.append(self.graph.node_inverted_index[j])
            if float(v) == float(0):
                neutral_values.append(self.graph.node_inverted_index[j])
        components.append(positive_values)
        components.append(negative_values)
        components.append(neutral_values)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # Fiedler, M. (1975). A property of eigenvectors of nonnegative symmetric matrices and
    # its application to graph theory. Czechoslovak mathematical journal, 25(4), 619-633.
    def akef_spectral_fiedler_connected_component_detection(self):
        components = []
        eigenvals, eigenvecs = scipy.linalg.eigh(self.graph.laplacian_weighted_enriched.todense())
        eigenvecs = eigenvecs.real
        eigenvals = eigenvals.real
        fiedler_pos = np.where(eigenvals == np.sort(eigenvals)[1])[0][0]
        fiedler_vector = np.real(np.transpose(eigenvecs)[fiedler_pos])
        fiedler_dict = {}
        for j, v in enumerate(fiedler_vector):
            fiedler_dict[self.graph.node_inverted_index[j]] = round(v, 10)
        fielder_grouped_sorted = defaultdict(list)
        for key, val in sorted(fiedler_dict.items()):
            fielder_grouped_sorted[val].append(key)
        for key, val in fielder_grouped_sorted.items():
            components.append(val)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # Fiedler, M. (1975). A property of eigenvectors of nonnegative symmetric matrices and
    # its application to graph theory. Czechoslovak mathematical journal, 25(4), 619-633.
    def akef_spectral_signed_fiedler_connected_component_detection(self):
        components = []
        eigenvals, eigenvecs = scipy.linalg.eigh(self.graph.laplacian_weighted_enriched.todense())
        eigenvecs = eigenvecs.real
        eigenvals = eigenvals.real
        fiedler_pos = np.where(eigenvals == np.sort(eigenvals)[1])[0][0]
        fiedler_vector = np.real(np.transpose(eigenvecs)[fiedler_pos])
        positive_values = {}
        negative_values = {}
        neutral_values = {}
        for j, v in enumerate(fiedler_vector):
            if float(v) > float(0):
                positive_values[self.graph.node_inverted_index[j]] = round(v, 5)
            if float(v) < float(0):
                negative_values[self.graph.node_inverted_index[j]] = round(v, 5)
            if float(v) == float(0):
                neutral_values[self.graph.node_inverted_index[j]] = round(v, 5)
        positive_reorganized = defaultdict(list)
        for key, val in sorted(positive_values.items()):
            positive_reorganized[val].append(key)
        negative_reorganized = defaultdict(list)
        for key, val in sorted(negative_values.items()):
            negative_reorganized[val].append(key)
        neutral_reorganized = defaultdict(list)
        for key, val in sorted(neutral_values.items()):
            neutral_reorganized[val].append(key)
        for k, v in positive_reorganized.items():
            components.append(v)
        for k, v in negative_reorganized.items():
            components.append(v)
        for k, v in neutral_reorganized.items():
            components.append(v)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list