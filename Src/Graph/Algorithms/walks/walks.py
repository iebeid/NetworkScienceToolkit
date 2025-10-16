import networkx as nx
import random
from models.data_utils import * 

def node2vec_walk(G, walk_length, start_walk,is_directed,p,q): 
    # print "start node: ", type(start_node), start_node
    '''
    return a random walk path, with walk array values alternating betwee nodes and edges.
    '''    
    walk = start_walk
    while len(walk) < walk_length:# here we may need to consider some dead end issues
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        random.shuffle(cur_nbrs)
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                rand = int(np.random.rand()*len(cur_nbrs))
                next_node = cur_nbrs[rand] 
                next_edge = G[cur][next_node]['type']
                walk.append(edge2str(next_edge))           
                walk.append(next_node)
            else:
                prev_node = walk[-3]
                pre_edge_type = G[prev_node][cur]['type']
                distance_sum = 0
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    # print "neighbor_link: ",neighbor_link
                    neighbor_link_type = neighbor_link['type']
                    # print "neighbor_link_type: ",neighbor_link_type
                    neighbor_link_weight = neighbor_link['weight']
                    
                    if G.has_edge(neighbor,prev_node) or G.has_edge(prev_node,neighbor):#undirected graph
                        
                        distance_sum += neighbor_link_weight/p #+1 normalization
                    elif neighbor == prev_node: #decide whether it can random walk back
                        distance_sum += neighbor_link_weight
                    else:
                        distance_sum += neighbor_link_weight/q

                '''
                pick up the next step link
                ''' 

                rand = np.random.rand() * distance_sum
                threshold = 0 
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    # print "neighbor_link: ",neighbor_link
                    neighbor_link_type = neighbor_link['type']
                    # print "neighbor_link_type: ",neighbor_link_type
                    neighbor_link_weight = neighbor_link['weight']                    
                    if G.has_edge(neighbor,prev_node) or G.has_edge(prev_node,neighbor):#undirected graph 
                        threshold += neighbor_link_weight/p 
                        if threshold >= rand:
                            next_node = neighbor
                    elif neighbor == prev_node:
                        threshold += neighbor_link_weight
                        if threshold >= rand:
                            next_node = neighbor
                    else:
                        threshold += neighbor_link_weight/q
                        if threshold >= rand:
                            next_node = neighbor
                next_edge = G[cur][next_node]['type']
                walk.append(edge2str(next_edge))
                walk.append(next_node) 
        else:
            break #if only has 1 neighbour 
 
        # print "walk length: ",len(walk),walk
        # print "edge walk: ",len(edge_walk),edge_walk 
    return walk  

def edge2vec_walk(G, walk_length, start_walk,is_directed,matrix,p,q): 
    # print "start node: ", type(start_node), start_node
    '''
    return a random walk path
    '''
    walk = start_walk
    while len(walk) < walk_length:# here we may need to consider some dead end issues
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) is 0:
            # Dead end. 
            break
        random.shuffle(cur_nbrs)
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                rand = int(np.random.rand()*len(cur_nbrs))
                next_node = cur_nbrs[rand] 
                next_edge = G[cur][next_node]['type']
                walk.append(edge2str(next_edge))           
                walk.append(next_node)
            else:
                prev_node = walk[-3]
                pre_edge_type = G[prev_node][cur]['type']
                distance_sum = 0
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    # print "neighbor_link: ",neighbor_link
                    neighbor_link_type = neighbor_link['type']
                    # print "neighbor_link_type: ",neighbor_link_type
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type-1][neighbor_link_type-1]
                    
                    if G.has_edge(neighbor,prev_node) or G.has_edge(prev_node,neighbor):#undirected graph
                        
                        distance_sum += trans_weight*neighbor_link_weight/p #+1 normalization
                    elif neighbor == prev_node: #decide whether it can random walk back
                        distance_sum += trans_weight*neighbor_link_weight
                    else:
                        distance_sum += trans_weight*neighbor_link_weight/q

                '''
                pick up the next step link
                ''' 

                rand = np.random.rand() * distance_sum
                threshold = 0 
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    # print "neighbor_link: ",neighbor_link
                    neighbor_link_type = neighbor_link['type']
                    # print "neighbor_link_type: ",neighbor_link_type
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type-1][neighbor_link_type-1]
                    
                    if G.has_edge(neighbor,prev_node) or G.has_edge(prev_node,neighbor):#undirected graph 
                        threshold += trans_weight*neighbor_link_weight/p 
                        if threshold >= rand:
                            next_node = neighbor
                    elif neighbor == prev_node:
                        threshold += trans_weight*neighbor_link_weight
                        if threshold >= rand:
                            next_node = neighbor
                    else:
                        threshold += trans_weight*neighbor_link_weight/q
                        if threshold >= rand:
                            next_node = neighbor
                next_edge = G[cur][next_node]['type']
                walk.append(edge2str(next_edge))
                walk.append(next_node) 
        else:
            break #if only has 1 neighbour 
 
        # print "walk length: ",len(walk),walk
        # print "edge walk: ",len(edge_walk),edge_walk 
    return walk  