from gensim import utils
import gensim.models
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


class MyCorpus(object):
    # class to stream the text data instead of loading it all in memory
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def __iter__(self):
        for line in open(self.corpus_file):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

def main():
    corpus = MyCorpus("metapaths_random_walks.txt")


    model = gensim.models.Word2Vec(sentences=corpus, corpus_file=None, size=128, alpha=0.05, window=5, workers=1, iter=500,
                                   min_alpha=0.0001, sg=1, hs=0, negative=10, compute_loss=True, sorted_vocab=1, min_count=1)

    # for word in model.wv.vocab:
    #     print(word)

    print("Size of vocab: " + str(len(model.wv.vocab)))

    # result = model.wv.most_similar(positive=['sun'], topn=20)

    # print(result)

    score = model.wv.evaluate_word_analogies("evaluation-samples.txt",dummy4unknown=True)

    print(score)

    vectors = []
    for row in model.wv.vectors:
        w_vector = list(row)
        vectors.append(w_vector)
    X = np.array(vectors)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], legend='full', palette='bright')
    plt.show()

if __name__ == "__main__":
    main()
