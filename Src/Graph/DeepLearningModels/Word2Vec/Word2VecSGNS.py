import numpy as np
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import sklearn
from numpy.linalg import norm
import random
import scipy


def remove_non_ascii(string):
    otptstr = ""
    for i in string:
        num = ord(i)
        if (num >= 0):
            if (num <= 127):
                otptstr = otptstr + i
    return otptstr


def preprocess_word(word):
    preprocessed_word = word.lower()
    processed_word = remove_non_ascii(preprocessed_word)
    return processed_word


def analyze_corpus(corpus, sample_rate=0.001, sample_threshold=0.2, factor_of_negative_sample=0.75):
    # tokenization and preparing a padded numpy matrix of words
    sizes_of_sentences = []
    number_of_documents = 0
    tokenized_corpus = []
    for document in corpus:
        number_of_documents = number_of_documents + 1
        words = document[0].split()
        size_of_current_sentence = len(words)
        sizes_of_sentences.append(size_of_current_sentence)
        words_list = []
        for word in words:
            word = preprocess_word(word)
            words_list.append(word)
        tokenized_corpus.append(words_list)
    sizes_of_sentences.sort(reverse=True)
    # finding the longest sentence to normalize and augment all sentences to the same size array
    longest_sentence = sizes_of_sentences[0]
    size_of_corpus = number_of_documents * longest_sentence
    # augment and pad numpy array with zeros
    for doc in tokenized_corpus:
        current_length_of_document = len(doc)
        if len(doc) < longest_sentence:
            augmentation_length = longest_sentence - current_length_of_document
            for i in range(augmentation_length):
                doc.append('0')
    # find unique vocab and add it to a global
    sentences_numpy_array = np.array(tokenized_corpus)
    unique, counts = np.unique(sentences_numpy_array, return_counts=True)
    occurances_dictionary = dict(zip(unique, counts))
    sorted_occurances_dictionary = sorted(occurances_dictionary.items(), key=lambda x: x[1], reverse=True)
    model_dataframe = pd.DataFrame.from_dict(sorted_occurances_dictionary)
    model_dataframe.columns = ["word", "count"]
    # down sampling probabilty of staying in training vocab
    list_of_z_of_word = []
    list_of_probablity_of_keeping_the_word = []
    for index1, row1 in model_dataframe.iterrows():
        count = row1[1]
        z_of_word = float(int(count) / int(size_of_corpus))
        probablity_of_removing_the_word = ((np.sqrt((z_of_word / sample_rate))) + 1) * (sample_rate / z_of_word)
        list_of_z_of_word.append(z_of_word)
        list_of_probablity_of_keeping_the_word.append(probablity_of_removing_the_word)
    model_dataframe["frequency"] = list_of_z_of_word
    model_dataframe["probability_of_staying"] = list_of_probablity_of_keeping_the_word
    # drop extra zero from augmentation and down sample vocab
    tokens_to_drop_from_corpus = []
    for index2, row2 in model_dataframe.iterrows():
        if row2["probability_of_staying"] <= sample_threshold:
            model_dataframe.drop(int(index2), axis=0, inplace=True)
            tokens_to_drop_from_corpus.append(str(row2["word"]))
    model_dataframe.reset_index(inplace=True, drop=True)
    tokenized_corpus = [[ele for ele in sub if ele not in tokens_to_drop_from_corpus] for sub in tokenized_corpus]
    # Load words into
    for index3, row3 in model_dataframe.iterrows():
        current_word = str(row3["word"])
        model_dataframe.at[index3, "word"] = current_word
    vocab = []
    for index, row in model_dataframe.iterrows():
        word = row["word"]
        vocab.append(word)
    # negative sampling
    train_words_pow = 0.0
    for index4, row4 in model_dataframe.iterrows():
        word_count = row4[1]
        train_words_pow += int(word_count) ** factor_of_negative_sample
    cumulative = 0.0
    list_of_cumulative_distribution_for_negative_sampling = []
    domain = len(model_dataframe.index)
    for index5, row5 in model_dataframe.iterrows():
        current_word_count = row5[1]
        cumulative = cumulative + (int(current_word_count) ** factor_of_negative_sample)
        cumulative_distribution_for_negative_sampling = round(cumulative / train_words_pow * domain)
        list_of_cumulative_distribution_for_negative_sampling.append(cumulative_distribution_for_negative_sampling)
    model_dataframe["cumulative_distribution"] = list_of_cumulative_distribution_for_negative_sampling
    return model_dataframe, tokenized_corpus, vocab


def get_word_one_hot_vector(model_dataframe, word):
    # one hot vector representation
    vocab_size = len(model_dataframe.index)
    one_hot_vector = np.array([])
    for index, row in model_dataframe.iterrows():
        temp = np.zeros(vocab_size)
        if word == str(row("word")):
            temp[index] = 1
        one_hot_vector = temp
    return one_hot_vector


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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return scipy.special.expit(x)


def cosine_similarity(vector1, vector2):
    similarity = float(float(1.0) - (
        scipy.spatial.distance.cosine(np.array(vector1, dtype=np.float), np.array(vector2, dtype=np.float))))
    return similarity


def visualize_embedding(model):
    vectors = []
    for index, row in enumerate(model):
        w_vector = list(row)
        vectors.append(w_vector)
    X = np.array(vectors)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], legend='full', palette='bright')
    plt.show()


def most_similar(word, top_n, model, model_dataframe):
    print(word)
    word_vector = model[get_word_index(model_dataframe, word)]
    scores_dictionary = {}
    scores = []
    for index, row in enumerate(model):
        w_vector = list(row)
        score = cosine_similarity(word_vector, w_vector)
        current_word = str(model_dataframe.at[index, "word"])
        current_dictionary = {float(score): current_word}
        scores_dictionary.update(current_dictionary)
        scores.append(score)
    scores.sort(reverse=True)
    similar_words = []
    for i in range(top_n):
        current_score = scores[i]
        similar_word = scores_dictionary.get(current_score)
        if word != similar_word:
            similar_words.append((similar_word, current_score))
            print(similar_word + " ---> " + str(current_score))
    return similar_words


def normalize_vector_l2(vector):
    return vector / norm(vector)


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


def generate_training_data_negative_sampling(model_dataframe, tokenized_corpus, window_size,
                                             negative_sampling_window_size):
    vocab_size = len(model_dataframe.index)
    training_data = []
    for document in tokenized_corpus:
        result = sliding_window(document, window_size)
        for context_window in result:
            middle_index = int(len(context_window) / 2)
            predict_word = context_window[middle_index]
            context_list = []
            for index, context_word in enumerate(context_window):
                if (index != middle_index) and (context_word != None):
                    negative_samples = []
                    for i in range(negative_sampling_window_size):
                        random_negative_index = random.sample(range(vocab_size), 1)
                        negative_samples_index = model_dataframe["cumulative_distribution"].searchsorted(
                            random_negative_index)
                        negative_sampled_word = model_dataframe.at[int(negative_samples_index), 'word']
                        negative_sample = str(negative_sampled_word)
                        negative_samples.append(negative_sample)
                    positive_sample = str(context_word)
                    context_list.append([positive_sample, negative_samples])
            if context_list:
                training_data.append([str(predict_word), context_list])
    return training_data


def get_word_index(model, word):
    target_word_index = model.index[model["word"] == word]
    return int(target_word_index.tolist()[0])


def train(corpus, window_size, embedding_dimension, number_of_epochs, learning_rate, min_learning_rate,
          negative_sampling_window_size,
          model_file):
    # Prepare learning rate
    decaying_learning_rate = np.linspace(learning_rate, min_learning_rate, num=number_of_epochs)
    # Prepare corpus
    model_dataframe, tokenized_corpus, vocab = analyze_corpus(corpus)
    negative_sampling_training_data = generate_training_data_negative_sampling(model_dataframe, tokenized_corpus,
                                                                               window_size,
                                                                               negative_sampling_window_size)
    # Initialize weight matrices
    vocab_size = len(model_dataframe.index)
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
        for sample in negative_sampling_training_data:
            # prepare input word
            predict_word = str(sample[0])
            context_samples = sample[1]
            index_of_predict_word = vocab.index(predict_word)
            # prepare output samples
            for output_words in context_samples:
                context_word = str(output_words[0])
                index_of_context_word = vocab.index(context_word)
                negative_samples = output_words[1]
                predict_word_label = [1]
                negative_sample_label = 0
                negative_sample_labels = []
                predict_word_index = [index_of_predict_word]
                negative_samples_indices = []
                for negative_sample in negative_samples:
                    negative_sample_index = vocab.index(negative_sample)
                    negative_samples_indices.append(negative_sample_index)
                    negative_sample_labels.append(negative_sample_label)
                sample_target_labels = np.array(predict_word_label + negative_sample_labels)
                samples_indices = predict_word_index + negative_samples_indices
                # Forward pass
                l1 = w1[index_of_context_word, :]
                l2 = w2[:, samples_indices]
                prod_term = np.dot(l1, l2)
                fb = sigmoid(prod_term)
                # Compute error
                e = (sample_target_labels - fb)
                g = e * decaying_learning_rate[i]
                error = error + np.sqrt(np.sum(e ** 2) / len(e))
                # Backpropagation and updating parameters
                dl_dw2 = np.outer(l1, g)
                dl_dw1 = np.dot(l2, g.T)
                # Update embedding
                w1[index_of_context_word, :] += dl_dw1
                w2[:, samples_indices] += dl_dw2
                # Compute loss
                loss -= sum(np.log(sigmoid(-1 * prod_term[1:])))  # for the context words
                loss -= np.log(sigmoid(prod_term[0]))  # for the predicted word
        if (i % 10 == 0):
            print(' EPOCH: ', i, ' | LOSS: ', np.round(loss, decimals=4), ' | RMSE: ',
                  np.round(error, decimals=4))
    # Normalize embedding
    w1 = sklearn.preprocessing.normalize(w1, norm="l2")
    with open("word2vec_model.npy", 'wb') as f:
        np.save(f, w1)
    model_dataframe.to_csv("model_dataframe.csv")


def log_evaluate_word_analogies(section):
    correct, incorrect = len(section['correct']), len(section['incorrect'])
    if correct + incorrect > 0:
        score = correct / (correct + incorrect)
        print("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
        return score


if __name__ == "__main__":
    window_size = 5
    embedding_dimension = 128
    number_of_epochs = 1000
    learning_rate = 0.05
    min_learning_rate = 0.0001
    negative_sampling_window_size = 10
    model_dataframe = "word2vec_model.csv"
    corpus = MyCorpus("corpus.txt")
    train(corpus, window_size, embedding_dimension, number_of_epochs, learning_rate, min_learning_rate,
          negative_sampling_window_size,
          model_dataframe)
    model = np.array([])
    with open("word2vec_model.npy", 'rb') as f:
        model = np.load(f, allow_pickle=True)
    model_dataframe = pd.read_csv("model_dataframe.csv")
    print("Size of vocab: " + str(len(model_dataframe.index)))
    similar_words = most_similar("sun", 20, model, model_dataframe)
    visualize_embedding(model)
