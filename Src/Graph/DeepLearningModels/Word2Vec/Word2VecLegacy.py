import numpy as np
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import scipy
from numpy.linalg import norm
import random


# Reduces the vocabulary by removing infrequent tokens
# The subsampling randomly discards frequent words while keeping the ranking same

def remove_non_ascii(string):
    otptstr = ""
    for i in string:
        num = ord(i)
        if (num >= 0):
            if (num <= 127):
                otptstr = otptstr + i
    return otptstr


def parse_word_label(labeled_word):
    raw_label = str(labeled_word[0:2])
    processed_label = str(labeled_word[1])
    pure_word = str(labeled_word[3:])
    return pure_word, processed_label


def preprocess_word(word):
    preprocessed_word = word.lower()
    processed_word = remove_non_ascii(preprocessed_word)
    return processed_word


def tokenize_corpus(corpus):
    list_of_sentences = []
    for document in corpus:
        words = document[0].split()
        words_list = []
        for word in words:
            word = preprocess_word(word)
            words_list.append(word)
        list_of_sentences.append(words_list)
    return list_of_sentences


def to_one_hot_vector(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


def sliding_window(iterable, n):
    if (n % 2 == 0):
        n = n + 1
    THIRD_WINDOW_SIZE = int((n - 1) / 2)
    iterables = itertools.tee(iterable, n)
    new_iterables_list = []
    for iterable, num_skipped in zip(iterables, itertools.count()):
        new_iterable = iter(([None] * THIRD_WINDOW_SIZE) + list(iterable) + ([None] * THIRD_WINDOW_SIZE))
        for _ in range(num_skipped):
            next(new_iterable, None)
        new_iterables_list.append(new_iterable)
    new_iterables = iter(new_iterables_list)
    return list(zip(*new_iterables))


def word_to_index(unique_vocab, word):
    index = -1
    for i, w in enumerate(unique_vocab):
        if w == word:
            index = i
    return index


def generate_training_data(WINDOW_SIZE, sentences, unique_vocab):
    training_data = []
    word2int = {}
    int2word = {}
    vocab_size = len(unique_vocab)
    for i, word in enumerate(unique_vocab):
        word2int[word] = i
        int2word[i] = word
    for sentence in sentences:
        result = sliding_window(sentence, WINDOW_SIZE)
        for context_window in result:
            context_window_list = []
            middle_index = int(len(context_window) / 2)
            middle_item = context_window[middle_index]
            for index, word in enumerate(context_window):
                if (index != middle_index) and (word != None):
                    context_window_list.append(to_one_hot_vector(word2int[word], vocab_size))
            training_data.append([to_one_hot_vector(word2int[middle_item], vocab_size), context_window_list])
    return training_data


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def cosine_similarity(vector1, vector2):
    similarity = float(1 - (scipy.spatial.distance.cosine(np.array(vector1), np.array(vector2))))
    return similarity


def visualize_embedding(model_dictionary):
    vectors = []
    for current_word in model_dictionary:
        w_vector = model_dictionary[current_word]
        vectors.append(w_vector)
    X = np.array(vectors)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], legend='full', palette='bright')
    plt.show()


def compute_word_analogies(a, b, c, expected, model):
    # b + c - a = expected
    error_threshold = 0.1
    first_opertaion = np.add(b, c)
    resultant_vector = np.subtract(a, first_opertaion)
    scores = []
    for current_word in model_dictionary:
        w_vector = model_dictionary[current_word]
        similarity_score = cosine_similarity(w_vector, resultant_vector)
        scores.append(similarity_score)
    scores.sort(reverse=True)
    if (float(scores[0]) <= 1.0) or (float(scores[0]) > float(1 - error_threshold)):
        return


def most_similar(word, top_n, model_dictionary):
    word_vector = model_dictionary.get(word)
    scores_dictionary = {}
    scores = []
    for current_word in model_dictionary:
        w_vector = model_dictionary[current_word]
        score = cosine_similarity(word_vector, w_vector)
        current_dictionary = {float(score): str(current_word)}
        scores_dictionary.update(current_dictionary)
        scores.append(score)
    scores.sort(reverse=True)
    similar_words = []
    for i in range(top_n):
        current_score = scores[i]
        similar_word = scores_dictionary.get(current_score)
        similar_words.append((similar_word, current_score))
    return similar_words


def normalize_vector_l2(vector):
    return vector / norm(vector)


def create_model_dictionary(unique_vocab, embedding_matrix):
    # create model dictionary
    model_dictionary = {}
    for word in unique_vocab:
        word_index = word_to_index(unique_vocab, word)
        word_vector = np.array(embedding_matrix[word_index])
        # normalize embedding vectors
        # normalized_word_vector_1 = (word_vector - np.min(word_vector)) / (np.max(word_vector) - np.min(word_vector))
        normalized_word_vector = normalize_vector_l2(word_vector)
        current_dictionary = {str(word): list(normalized_word_vector)}
        model_dictionary.update(current_dictionary)
    return model_dictionary


def unique(list):
    return [i for n, i in enumerate(list) if i not in list[:n]]


def sentences_to_unique_vocab(sentences):
    unique_vocab = []
    for sentence in sentences:
        sentence_unique_words_list = sentence
        unique_vocab = unique_vocab + sentence_unique_words_list
    unique_vocab = unique(unique_vocab)
    return unique_vocab


class MyCorpus(object):
    # class to stream the text data instead of loading it all in memory
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def simple_preprocess(self, document):
        tokens = [document]
        return tokens

    def __iter__(self):
        for line in open(self.corpus_file):
            # assume there's one document per line, tokens separated by whitespace
            yield self.simple_preprocess(line)


def create_vocab(tokenized_corpus):
    unique_vocab = []
    for sentence in tokenized_corpus:
        sentence_unique_words_list = sentence
        unique_vocab = unique_vocab + sentence_unique_words_list
    unique_vocab = unique(unique_vocab)
    return unique_vocab


def train(corpus, window_size, embedding_dimension, number_of_epochs, learning_rate, negative_sampling_window_size):
    tokenized_corpus = tokenize_corpus(corpus)
    unique_vocab = sentences_to_unique_vocab(tokenized_corpus)
    training_data = generate_training_data(window_size, tokenized_corpus, unique_vocab)
    # Initialize weight matrices
    vocab_size = len(unique_vocab)
    # Initialization based on https://arxiv.org/pdf/1711.09160.pdf (Glorot and Bengio (2010))
    low = -((np.sqrt(6)) / (np.sqrt((vocab_size + embedding_dimension))))
    high = ((np.sqrt(6)) / (np.sqrt((vocab_size + embedding_dimension))))
    w1 = np.random.uniform(low, high, (vocab_size, embedding_dimension))  # embedding matrix
    w2 = np.random.uniform(low, high, (embedding_dimension, vocab_size))  # context matrix
    # loop on epocs
    for i in range(0, number_of_epochs):
        loss = 0
        error = 0
        # loop on each training sample
        for sample in training_data:
            input_word = sample[0]
            input_word_index = word_to_index(unique_vocab,input_word)
            input_word_one_hot_vector = to_one_hot_vector(input_word_index,vocab_size)
            context_window = sample[1]
            # do a forward pass
            h = np.dot(w1.T, np.array(input_word_one_hot_vector))
            u = np.dot(w2.T, h)
            y_prediction = softmax(u.T)
            # calculate error
            # [(float(target_word[1]) - float(y_prediction[get_word_index(model_dataframe, str(target_word[0]))]))
            error = np.sum([np.subtract(y_prediction, context_word[1]) for context_word in context_window], axis=0)
            print(error.shape)

            # negative sampling backpropagation
            dl_dw2 = np.outer(h, error)
            # print(dl_dw2.shape)

            dl_dw1 = np.outer(input_word_one_hot_vector, np.dot(w2, error.T))
            print(w2.shape)

            # #-------------------------------------
            # #update W1
            # for index, context_word in enumerate(context_window):
            #     context_word_index = get_word_index(model_dataframe, str(context_word[0]))
            #     w1[context_word_index,:] = w1[context_word_index,:] - ((w2[:,context_word_index] * error[:,context_word_index]) * learning_rate)
            #
            #
            # #update W2
            # for index, context_word in enumerate(context_window):
            #     context_word_index = get_word_index(model_dataframe, str(context_word[0]))
            #     # w2[:,context_word_index] = w2[:,context_word_index] - (learning_rate * (w1[context_word_index,:] * error[:,context_word_index]))
            #     w2[:, context_word_index] = w2[:, context_word_index] - np.dot(np.dot(h, error[:, context_word_index]), learning_rate)
            # #-------------------------------------------

            # update

            # # negative sampling update weight matrix
            # w1[input_word_index] = w1[input_word_index] - (learning_rate * dl_dw1[input_word_index]) # we update only the positive words
            # # we update both positive and negative samples
            # for context_word in context_window:
            #     index_of_current_context_word = get_word_index(model_dataframe,str(context_word[0]))
            #     w2[index_of_current_context_word] = w2[index_of_current_context_word] - (learning_rate * dl_dw2[:][index_of_current_context_word])

            w1 = w1 - (learning_rate * dl_dw1)
            w2 = w2 - (learning_rate * dl_dw2)

            # https://ruder.io/word-embeddings-softmax/index.html#negativesampling
            # calculate loss
            # postive_samples_term = np.log(sigmoid(u))
            # negative_samples_term = np.sum([np.log(sigmoid(-1*u[negative_sample])) for negative_sample in negative_samples])
            # loss -= np.sum((postive_samples_term+negative_samples_term))
            loss += -np.sum(
                [u[get_word_one_hot_vector(model_dataframe, context_word[0]).tolist().index(1)] for context_word in
                 context_window]) + len(context_window) * np.log(np.sum(np.exp(u)))
        print(' EPOCH: ', i, ' LOSS: ', loss, ' ROOT MEAN SQUARE ERROR: ', np.sqrt(np.mean(error) ** 2))

    model_dictionary = create_model_dictionary(unique_vocab, w1)
    print(model_dictionary)
    return model_dictionary


def log_evaluate_word_analogies(section):
    correct, incorrect = len(section['correct']), len(section['incorrect'])
    if correct + incorrect > 0:
        score = correct / (correct + incorrect)
        print("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
        return score


# def evaluate_word_analogies(analogies,vocab):
#     #https://aclweb.org/aclwiki/Analogy_(State_of_the_art)
#     sections, section = [], None
#     quadruplets_no = 0
#     for line in open(analogies):
#         if line.startswith(': '):
#             # a new section starts => store the old section
#             if section:
#                 sections.append(section)
#                 log_evaluate_word_analogies(section)
#             section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
#         else:
#             if not section:
#                 raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
#             try:
#                 if case_insensitive:
#                     a, b, c, expected = [word.upper() for word in line.split()]
#                 else:
#                     a, b, c, expected = [word for word in line.split()]
#             except ValueError:
#                 print("Skipping invalid line #%i in %s", line_no, analogies)
#                 continue
#             quadruplets_no += 1


if __name__ == "__main__":
    # corpus = [
    #     ['The Solar System is the gravitationally bound system of the Sun and the objects that orbit it either directly or indirectly'],
    #     ['Of the objects that orbit the Sun directly the largest are the eight planets with the remainder being smaller objects the dwarf planets and small Solar System bodies'],
    #     ['Of the objects that orbit the Sun indirectly like the moons two are larger than the smallest planet Mercury'],
    #     ['The Solar System formed 4.6billion years ago from the gravitational collapse of a giant interstellar molecular cloud'],
    #     ['The vast majority of the system mass is in the Sun with the majority of the remaining mass contained in Jupiter'],
    #     ['The four smaller inner planets Mercury Venus Earth and Mars are terrestrial planets being primarily composed of rock and metal'],
    #     ['The four outer planets are giant planets being substantially more massive than the terrestrials'],
    #     ['The two largest Jupiter and Saturn are gas giants being composed mainly of hydrogen and helium'],
    #     ['The two outermost planets Uranus and Neptune are ice giants being composed mostly of substances with relatively high melting points compared with hydrogen and helium called volatiles such as water ammonia and methane'],
    #     ['All eight planets have almost circular orbits that lie within a nearly flat disc called the ecliptic']
    # ]
    window_size = 3
    embedding_dimension = 24
    number_of_epochs = 1000
    learning_rate = 0.25
    negative_sampling_window_size = 5

    corpus = MyCorpus("corpus-annotated.txt")

    model_dictionary = train(corpus, window_size, embedding_dimension, number_of_epochs, learning_rate,
                             negative_sampling_window_size)

    most_similar("sun", 10, model_dictionary)

    visualize_embedding(model_dictionary)
