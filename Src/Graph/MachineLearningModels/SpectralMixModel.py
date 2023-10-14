import tensorflow as tf

class SpectralMixModel:
    def __init__(self, graph, d):
        self.graph = graph
        self.d = d
        self.N = self.graph.N
        self.R = len(self.graph.relation_adjacency_matrices.keys())
        categories = []
        for k,v in self.graph.node_attributes.items():
            print(k, v)
            for ci in v:
                categories.append(ci)
        categories=list(set(categories))
        self.C = len(categories)
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.O=tf.Variable(self.initializer(shape=(self.N, self.d)), name="O")
        self.M=tf.Variable(self.initializer(shape=(self.C, self.d)), name="M")

    def train(self):
        for i in range(self.N):
            for r in range(self.R):
                alpha_r = 0
                N_r = self.graph.get_relational_neighbors(r,i)
                for p in range(len(N_r)):
                    for l in range(self.d):
                        o_il=o_il+alpha_r*w_ip*o_pl/self.graph.get_relational_neighbors(i,r)


            