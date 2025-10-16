import pickle
import copy
import itertools
import pandas as pd
import sklearn
import random
import scipy
from tqdm import tqdm
import time
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.spatial.distance import cosine
from queue import Queue
import threading


class MyCorpus(object):
    # class to stream the text data instead of loading it all in memory
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    @staticmethod
    def simple_preprocess(document):
        tokens = document.strip().split()
        return tokens

    def __iter__(self):
        for line in open(self.corpus_file):
            # assume there's one document per line, tokens separated by whitespace
            yield self.simple_preprocess(line)


class Chem2Bio2RDF(object):
    def __init__(self, nodes_file=""):
        print("Initialize Dataset")
        self.nodes_file = nodes_file

    @staticmethod
    def map_to_letter(number):
        map_value = None
        if number == '0':
            map_value = 'a'
        elif number == '1':
            map_value = 'b'
        elif number == '2':
            map_value = 'c'
        elif number == '3':
            map_value = 'd'
        elif number == '4':
            map_value = 'e'
        elif number == '5':
            map_value = 'f'
        elif number == '6':
            map_value = 'g'
        elif number == '7':
            map_value = 'h'
        elif number == '8':
            map_value = 'i'
        elif number == '9':
            map_value = 'j'
        return map_value

    @staticmethod
    def map_to_number(number):
        map_value = None
        if number == 'a':
            map_value = '0'
        elif number == 'b':
            map_value = '1'
        elif number == 'c':
            map_value = '2'
        elif number == 'd':
            map_value = '3'
        elif number == 'e':
            map_value = '4'
        elif number == 'f':
            map_value = '5'
        elif number == 'g':
            map_value = '6'
        elif number == 'h':
            map_value = '7'
        elif number == 'i':
            map_value = '8'
        elif number == 'j':
            map_value = '9'
        return map_value

    def map_code_to_letter(self, code):
        map_value = ''
        for c in code:
            single_map_value = self.map_to_letter(c)
            map_value = map_value + str(single_map_value)
        return map_value

    def map_code_to_number(self, code):
        map_value = ''
        for c in code:
            single_map_value = self.map_to_number(c)
            map_value = map_value + str(single_map_value)
        return int(map_value)

    @staticmethod
    def get_node_type(node):
        return node[0:2]


class Metapath2Vec(object):

    def __init__(self, walks, dataset, window_size, embedding_dimension, number_of_epochs, learning_rate,
                 min_learning_rate,
                 negative_sampling_window_size, sample_rate, sample_threshold, factor_of_negative_sample, workers,
                 batch_size, metapath2vec, pp, gensim,
                 model_file, gensim_model_file):
        self.walks = walks
        self.dataset = dataset
        self.window_size = window_size
        self.embedding_dimension = embedding_dimension
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.negative_sampling_window_size = negative_sampling_window_size
        self.sample_rate = sample_rate
        self.sample_threshold = sample_threshold
        self.factor_of_negative_sample = factor_of_negative_sample
        self.model_file = model_file
        self.gensim_model_file = gensim_model_file
        self.metapath2vec = metapath2vec
        self.pp = pp
        self.gensim = gensim
        # Prepare learning rate
        self.decaying_learning_rate = np.linspace(learning_rate, min_learning_rate, num=number_of_epochs)
        # Corpus variables
        self.vocab = []
        self.node_type_lists = []
        self.tokenized_corpus = []
        self.vocab_size = 0
        self.number_of_sentences = 0
        self.number_of_words = 0
        self.model_dataframe = None
        # Training data
        self.training_data = []
        self.workers = workers
        self.batch_size = batch_size
        self.queue_factor = 2
        self.w1 = None
        self.w2 = None
        self.error = 0
        self.loss = 0

    @staticmethod
    def remove_non_ascii(string):
        otptstr = ""
        for i in string:
            num = ord(i)
            if num >= 0:
                if num <= 127:
                    otptstr = otptstr + i
        return otptstr

    def preprocess_word(self, word):
        preprocessed_word = word.lower()
        processed_word = self.remove_non_ascii(preprocessed_word)
        return processed_word

    def analyze_corpus(self):
        # tokenization and preparing a padded numpy matrix of words
        print("Tokenization and preparing a padded numpy matrix of words ...")
        sizes_of_sentences = []

        for sentence in tqdm(self.walks):
            self.number_of_sentences = self.number_of_sentences + 1
            size_of_current_sentence = len(sentence)
            sizes_of_sentences.append(size_of_current_sentence)
            words_list = []
            for word in sentence:
                self.number_of_words = self.number_of_words + 1
                word = self.preprocess_word(word)
                words_list.append(word)
            self.tokenized_corpus.append(words_list)
        sizes_of_sentences.sort(reverse=True)
        # finding the longest sentence to normalize and augment all sentences to the same size array
        print("Finding the longest sentence to normalize and augment all sentences to the same size array ...")
        longest_sentence = sizes_of_sentences[0]
        # augment and pad numpy array with zeros
        print("Augmenting and pad numpy array with zeros ...")
        copy_of_tokenized_corpus = copy.deepcopy(self.tokenized_corpus)
        for doc in tqdm(copy_of_tokenized_corpus):
            current_length_of_document = len(doc)
            if len(doc) < longest_sentence:
                augmentation_length = longest_sentence - current_length_of_document
                for i in range(augmentation_length):
                    doc.append('0')
        # find unique vocab and add it to a global
        print("Finding unique vocab and add it to a global ...")
        sentences_numpy_array = np.array(copy_of_tokenized_corpus)
        unique, counts = np.unique(sentences_numpy_array, return_counts=True)
        occurances_dictionary = dict(zip(unique, counts))
        sorted_occurances_dictionary = sorted(occurances_dictionary.items(), key=lambda x: x[1], reverse=True)
        self.model_dataframe = pd.DataFrame(sorted_occurances_dictionary, columns=["word", "count"])
        self.model_dataframe.drop(int(0), axis=0, inplace=True)
        print("Extracting vocab ...")
        for index, row in self.model_dataframe.iterrows():
            word = row["word"]
            self.vocab.append(word)
        # negative sampling
        print("Preparing negative sampling ...")
        train_words_pow = 0.0
        for index4, row4 in self.model_dataframe.iterrows():
            word_count = row4[1]
            train_words_pow += int(word_count) ** self.factor_of_negative_sample
        cumulative = 0.0
        list_of_cumulative_distribution_for_negative_sampling = []
        self.vocab_size = int(len(self.vocab))
        print("Computing cumulative distribution ...")
        for index5, row5 in self.model_dataframe.iterrows():
            current_word_count = row5[1]
            cumulative = cumulative + (current_word_count ** self.factor_of_negative_sample)
            cumulative_distribution_for_negative_sampling = round(cumulative / train_words_pow * self.vocab_size)
            list_of_cumulative_distribution_for_negative_sampling.append(cumulative_distribution_for_negative_sampling)
        self.model_dataframe["cumulative_distribution"] = list_of_cumulative_distribution_for_negative_sampling
        # Extract word types
        word_types = []
        print("Extracting node types ...")
        for index66, row66 in self.model_dataframe.iterrows():
            current_word = str(row66["word"])
            word_types.append(self.dataset.get_node_type(current_word))
        self.model_dataframe["word_type"] = word_types
        node_types_unique = self.model_dataframe["word_type"].unique()
        print("Creating node type list ...")
        for node_type in tqdm(node_types_unique):
            node_type_list = []
            for index11, row11 in self.model_dataframe.iterrows():
                if row11["word_type"] == node_type:
                    node_type_list.append(row11["word"])
            self.node_type_lists.append((node_type, node_type_list))

    @staticmethod
    def sliding_window(iterable, n):
        if n % 2 == 0:
            n = n + 1
        third_window_size = int((n - 1) / 2)
        iterables = itertools.tee(iterable, n)
        new_iterables_list = []
        for iterable, num_skipped in zip(iterables, itertools.count()):
            new_iterable = iter(([None] * third_window_size) + list(iterable) + ([None] * third_window_size))
            for _ in range(num_skipped):
                next(new_iterable, None)
            new_iterables_list.append(new_iterable)
        new_iterables = iter(new_iterables_list)
        return list(zip(*new_iterables))

    def generate_train_data(self):
        heterogenous_dataframes = []
        for node_type_list in self.node_type_lists:
            heterogenous_dataframe = self.model_dataframe.loc[self.model_dataframe["word_type"] == node_type_list[0]]
            heterogenous_dataframe["count"].sum(skipna=True)
            heterogenous_dataframe.reset_index(drop=True, inplace=True)
            train_words_pow = 0.0
            for index4, row4 in heterogenous_dataframe.iterrows():
                train_words_pow += row4["count"] ** self.factor_of_negative_sample
            cumulative = 0.0
            domain = len(heterogenous_dataframe.index)
            for index5, row5 in heterogenous_dataframe.iterrows():
                cumulative = cumulative + (row5["count"] ** self.factor_of_negative_sample)
                heterogenous_dataframe.loc[index5, "heterogenous_cumulative_distribution"] = int(
                    round(cumulative / train_words_pow * domain) - 1)
            heterogenous_dataframes.append(heterogenous_dataframe)
        for document in tqdm(self.tokenized_corpus):
            result_window = self.sliding_window(document, self.window_size)
            for context_window in result_window:
                middle_index = int(len(context_window) / 2)
                predict_word = context_window[middle_index]
                context_list = []
                for index, context_word in enumerate(context_window):
                    if (index != middle_index) and (context_word is not None):
                        negative_samples = []
                        context_word_node_type = context_word[0:2]
                        for i in range(self.negative_sampling_window_size):
                            if self.pp:
                                for r, node_type_list in enumerate(self.node_type_lists):
                                    if context_word_node_type == node_type_list[0]:
                                        current_node_type_vocab_list = node_type_list[1]
                                        chosen_random_word = random.choice(current_node_type_vocab_list)
                                        heterogenous_model_dataframe = heterogenous_dataframes[r]
                                        chosen_random_word_index = int(heterogenous_model_dataframe.index[
                                                                           heterogenous_model_dataframe[
                                                                               "word"] == chosen_random_word].tolist()[
                                                                           0])
                                        # Heterogenous negative sampling
                                        negative_sample_index = heterogenous_model_dataframe[
                                            "heterogenous_cumulative_distribution"].searchsorted(
                                            chosen_random_word_index)
                                        negative_sample_word = heterogenous_model_dataframe.loc[
                                            int(negative_sample_index), "word"]
                                        negative_samples.append(str(negative_sample_word))
                            else:
                                random_negative_index = int(random.sample(range(self.vocab_size), 1)[0])
                                negative_samples_index = self.model_dataframe["cumulative_distribution"].searchsorted(
                                    random_negative_index)
                                negative_sampled_word = self.model_dataframe.at[negative_samples_index + 1, "word"]
                                negative_samples.append(str(negative_sampled_word))
                        context_list.append([context_word, negative_samples])
                if context_list:
                    self.training_data.append([str(predict_word), context_list])

    def get_index(self, word):
        for i in range(len(self.vocab)):
            if self.vocab[i] == word:
                yield i

    def get_vector(self, word):
        return np.array(self.w1[int(list(self.get_index(word))[0])])

    @staticmethod
    def sigmoid(x):
        return expit(x)

    def normalize(self):
        self.w1 = sklearn.preprocessing.normalize(self.w1, norm="l2")

    @staticmethod
    def cosine_similarity(vector1, vector2):
        similarity = float(float(1.0) - (
            scipy.spatial.distance.cosine(np.array(vector1, dtype=np.float), np.array(vector2, dtype=np.float))))
        return similarity

    def most_similar_word(self, word, top_n):
        scores_dictionary = {}
        scores = []
        for h in self.w1:
            print(word)
            score = self.cosine_similarity(self.get_vector(word), np.array(h))
            scores_dictionary.update({float(score): h["node_code"]})
            scores.append(score)
        scores.sort(reverse=True)
        similar_words = []
        for i in range(top_n):
            current_score = scores[i]
            similar_word = scores_dictionary.get(current_score)
            if word != similar_word:
                similar_words.append((similar_word, current_score))
        return similar_words

    @staticmethod
    def visualize_embedding(mode_file):
        loaded_model = pickle.load(open(mode_file, "rb"))
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        vectors = []
        types = []
        i = 0
        for index, row in enumerate(loaded_model):
            vectors.append(list(row["node_vector"]))
            types.append(row["node_type"])
            i = i + 1
        x = np.array(vectors)
        t = np.array(types)
        tsne = TSNE()
        x_embedded = tsne.fit_transform(x)
        sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=t, legend='full', palette='bright')
        plt.show()

    def save_model(self):
        m = []
        for index, record in self.model_dataframe.iterrows():
            word = str(record["word"])
            word_type = str(record["word_type"])
            word_vector = self.get_vector(word)
            word_record = {'node': word, 'node_type': word_type, 'node_vector': np.array(word_vector)}
            m.append(word_record)
        pickle.dump(m, open(self.model_file, "wb"))

    def convert_gensim_model_to_local_model(self, gensim_model_filename):
        gensim_model = Word2Vec.load(gensim_model_filename, mmap='r')
        ml = []
        for node in gensim_model.wv.vocab:
            node_vector = gensim_model.wv[node]
            node_record = {'node': str(node), 'node_type': str(self.dataset.get_node_type(node)),
                           'node_vector': np.array(node_vector)}
            ml.append(node_record)
        pickle.dump(ml, open(gensim_model_filename, "wb"))

    def train_pair(self, i, predict_word, output_words):
        context_word = output_words[0]
        index_of_context_word = int(list(self.get_index(context_word))[0])
        negative_samples = output_words[1]
        predict_node_label = [1]
        negative_sample_label = 0
        negative_sample_labels = []
        predict_word_index = [int(list(self.get_index(predict_word))[0])]
        negative_samples_indices = []
        for negative_sample in negative_samples:
            negative_sample_index = int(list(self.get_index(negative_sample))[0])
            negative_samples_indices.append(negative_sample_index)
            negative_sample_labels.append(negative_sample_label)
        sample_target_labels = np.array(predict_node_label + negative_sample_labels)
        samples_indices = predict_word_index + negative_samples_indices
        # Forward pass
        l1 = self.w1[index_of_context_word, :]
        l2 = self.w2[:, samples_indices]
        prod_term = np.dot(l1, l2)
        fb = self.sigmoid(prod_term)
        # Compute error
        g = (sample_target_labels - fb)
        e = g * self.decaying_learning_rate[i]
        self.error = self.error + np.sqrt(np.sum(g ** 2) / len(g))
        # Backpropagation
        dl_dw2 = np.outer(l1, e)
        dl_dw1 = np.dot(l2, e.T)
        # Update embedding
        self.w1[index_of_context_word, :] += dl_dw1
        self.w2[:, samples_indices] += dl_dw2
        # Compute loss
        self.loss -= sum(np.log(self.sigmoid(-1 * prod_term[1:])))  # for the context words
        self.loss -= np.log(self.sigmoid(prod_term[0]))  # for the predicted word

    def train_word(self, i, word):
        predict_word = word[0]
        context_samples = word[1]
        # prepare output samples
        for output_words in context_samples:
            self.train_pair(i, predict_word, output_words)

    def train_epoch(self, i):
        for word in self.training_data:
            self.train_word(i, word)

    def train_model_threaded(self):
        # Initialize weight matrices based on https://arxiv.org/pdf/1711.09160.pdf (Glorot and Bengio (2010))
        high = ((np.sqrt(6)) / (np.sqrt((self.vocab_size + self.embedding_dimension))))
        low = -high
        self.w1 = np.random.uniform(low, high, (self.vocab_size, self.embedding_dimension))  # embedding matrix
        self.w2 = np.random.uniform(low, high, (self.embedding_dimension, self.vocab_size))  # context matrix
        for i in range(0, self.number_of_epochs):
            job_queue = Queue(maxsize=self.queue_factor * self.workers)
            progress_queue = Queue(maxsize=(self.queue_factor + 1) * self.workers)

            workers = [
                threading.Thread(
                    target=self._worker_loop,
                    args=(job_queue, progress_queue,))
                for _ in range(self.workers)
            ]

            workers.append(threading.Thread(
                target=self._job_producer,
                args=(self.walks, job_queue),
                kwargs={'cur_epoch': i, 'total_examples': self.number_of_sentences,
                        'total_words': self.number_of_words}))

            for thread in workers:
                thread.daemon = True  # make interrupting the process with ctrl+c easier
                thread.start()

    def train_model(self):
        # Initialize weight matrices based on https://arxiv.org/pdf/1711.09160.pdf (Glorot and Bengio (2010))
        high = ((np.sqrt(6)) / (np.sqrt((self.vocab_size + self.embedding_dimension))))
        low = -high
        self.w1 = np.random.uniform(low, high, (self.vocab_size, self.embedding_dimension))  # embedding matrix
        self.w2 = np.random.uniform(low, high, (self.embedding_dimension, self.vocab_size))  # context matrix
        for i in range(0, self.number_of_epochs):
            self.loss = 0
            self.error = 0
            self.train_epoch(i)
            if i % 10 == 0:
                print(' EPOCH: ' + str(i) + ' | LOSS: ' + str(np.round(self.loss, decimals=4)) + ' | RMSE: ' + str(
                    np.round(self.error, decimals=4)))
        print(str(self.vocab_size))

    @staticmethod
    def _raw_word_count(job):
        return sum(len(sentence) for sentence in job)

    @staticmethod
    def _do_train_job():
        print("Implement threads for learning")
        return 0, 0

    def _worker_loop(self, job_queue, progress_queue):

        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            data_iterable, job_parameters = job

            tally, raw_tally = self._do_train_job()

            progress_queue.put((len(data_iterable), tally, raw_tally))  # report back progress
            jobs_processed += 1

    def _get_job_params(self, i):
        learning_rate = self.learning_rate - (
                (self.learning_rate - self.min_learning_rate) * float(i) / self.number_of_epochs)
        return learning_rate

    def _job_producer(self, i, data_iterator, job_queue):
        job_batch, batch_size = [], 0
        pushed_words, pushed_examples = 0, 0
        next_job_params = self._get_job_params(i)
        job_no = 0

        for data_idx, data in enumerate(data_iterator):
            data_length = self._raw_word_count([data])

            # can we fit this sentence into the existing job batch?
            if batch_size + data_length <= self.batch_size:
                # yes => add it to the current job
                job_batch.append(data)
                batch_size += data_length
            else:
                job_no += 1
                job_queue.put((job_batch, next_job_params))

                pushed_words += self._raw_word_count(job_batch)
                epoch_progress = 1.0 * pushed_words / self.number_of_words
                start_alpha = self.learning_rate
                end_alpha = self.min_learning_rate
                progress = (i + epoch_progress) / self.number_of_epochs
                next_alpha = start_alpha - (start_alpha - end_alpha) * progress
                next_alpha = max(end_alpha, next_alpha)
                self.min_alpha_yet_reached = next_alpha

                # add the sentence that didn't fit as the first item of a new job
                job_batch, batch_size = [data], data_length
        # add the last job too (may be significantly smaller than batch_words)
        if job_batch:
            job_no += 1
            job_queue.put((job_batch, next_job_params))

        # give the workers heads up that they can finish -- no more work!
        for _ in range(self.workers):
            job_queue.put(None)

    def train(self):
        if self.metapath2vec:
            start1 = time.time()
            self.analyze_corpus()
            print("Generating training data ...")
            self.generate_train_data()
            print("Start training Skip-gram with negative sampling ...")
            if self.workers > 1:
                self.train_model_threaded()
            else:
                self.train_model()
            self.normalize()
            end1 = time.time()
            print("Metapath2Vec Timing")
            print(end1 - start1)
            self.save_model()
            # self.visualize_embedding(self.model_file)
        if self.gensim:
            start2 = time.time()
            gensim_model = Word2Vec(sentences=self.walks, size=self.embedding_dimension, alpha=self.learning_rate,
                                    window=self.window_size,
                                    workers=self.workers, iter=self.number_of_epochs,
                                    min_alpha=self.min_learning_rate, sg=1, hs=0,
                                    negative=self.negative_sampling_window_size,
                                    sample=0, compute_loss=True,
                                    sorted_vocab=1, min_count=0)
            end2 = time.time()
            print("Gensim Word2Vec Timing")
            print(end2 - start2)
            print(gensim_model)
            gensim_model.save(self.gensim_model_file)
            self.convert_gensim_model_to_local_model(self.gensim_model_file)
            # self.visualize_embedding(self.gensim_model_file)


class Chem2Bio2RDF2Evaluation(object):
    def __init__(self, hits, evaluation_set_file, model_file, metapath2vec):
        print("Evaluate embedding model")
        self.model = pickle.load(open(model_file, "rb"))
        self.metpath2vec = metapath2vec
        self.hits = hits
        self.evaluation_set = pickle.load(open(evaluation_set_file, "rb"))
        self.vocab_size = len(self.model)
        self.filtered_evaluation_set = []
        self.left_node_types = ["dr", "go", "di", "co", "ti", "pa"]
        self.right_node_types = ["ta"]

    def check_node_in_vocab(self, word):
        for i, record in enumerate(self.model):
            node = record["node"]
            if node == word:
                return True

    def filter_evaluation_set(self):
        for row in tqdm(self.evaluation_set):
            expected = row["expected_value"]
            a_code = row["left_node_code"]
            b_code = row["right_node_code"]
            if self.check_node_in_vocab(a_code) and self.check_node_in_vocab(b_code):
                self.filtered_evaluation_set.append(
                    {"left": str(a_code), "right": str(b_code), "expected": str(expected)})

    def get_vector(self, word):
        for i, record in enumerate(self.model):
            if record["node"] == word:
                return np.array(record["node_vector"])

    @staticmethod
    def cosine_similarity(vector1, vector2):
        similarity = float(float(1.0) - (
            scipy.spatial.distance.cosine(np.array(vector1, dtype=np.float), np.array(vector2, dtype=np.float))))
        return similarity

    def most_similar_word(self, word, top_n):
        scores_dictionary = {}
        scores = []
        for i, record in enumerate(self.model):
            score = self.cosine_similarity(self.get_vector(word), np.array(record["node_vector"]))
            scores_dictionary.update({float(score): record["node"]})
            scores.append(score)
        scores.sort(reverse=True)
        similar_words = []
        for i in range(top_n):
            current_score = scores[i]
            similar_word = scores_dictionary.get(current_score)
            if word != similar_word:
                similar_words.append((similar_word, current_score))
        return similar_words

    def most_similar_edge(self, node, hit):
        similar_nodes = self.most_similar_word(node, self.vocab_size - 1)
        node_type = node[0:2]
        similar_left = []
        similar_right = []
        filtered_similar_left = []
        filtered_similar_right = []
        for similar_node in similar_nodes:
            similar_node = similar_node[0]
            similar_node_type = similar_node[0:2]
            if similar_node_type in self.left_node_types:
                similar_left.append(similar_node)
            if similar_node_type in self.right_node_types:
                similar_right.append(similar_node)
        for i in range(hit):
            if node_type in self.left_node_types:
                filtered_similar_right.append(similar_right[i])
            if node_type in self.right_node_types:
                filtered_similar_left.append(similar_left[i])
        if node_type in self.left_node_types:
            return filtered_similar_right
        if node_type in self.right_node_types:
            return filtered_similar_left

    def evaluate(self):
        true_positive = []
        false_negative = []
        true_negative = []
        false_positive = []

        self.filter_evaluation_set()
        for hit in tqdm(self.hits):
            for row in self.filtered_evaluation_set:
                expected = int(row["expected"])
                a_code = row["left"]
                b_code = row["right"]
                a_type = a_code[0:2]
                b_type = b_code[0:2]
                if a_type in self.left_node_types and b_type in self.right_node_types:
                    result_a = self.most_similar_edge(a_code, hit)
                    result_b = self.most_similar_edge(b_code, hit)
                    if expected == 1:
                        if (b_code in result_a) or (a_code in result_b):
                            true_positive.append(1)
                        else:
                            false_negative.append(1)
                    elif expected == 0:
                        if (b_code in result_a) or (a_code in result_b):
                            false_positive.append(1)
                        else:
                            true_negative.append(1)
            true_positive_count = len(true_positive)
            false_negative_count = len(false_negative)
            true_negative_count = len(true_negative)
            false_positive_count = len(false_positive)
            if true_positive_count != 0:
                f1 = float(
                    true_positive_count / (true_positive_count + 0.5 * (false_positive_count + false_negative_count)))
                result = "F1 Accuracy Score @ " + str(hit) + " is " + str(f1)
                print(result)


if __name__ == "__main__":
    param_window_size = 3
    param_embedding_dimension = 300
    param_number_of_epochs = 300
    param_learning_rate = 0.05
    param_min_learning_rate = 0.001
    param_negative_sampling_window_size = 5
    param_sample_rate = 0.001
    param_sample_threshold = 0.00001
    param_factor_of_negative_sample = 0.75
    param_workers = 1
    param_batch_size = 32
    param_hits = [10, 25, 50, 100]
    param_metapath2vec = True
    param_pp = True
    param_gensim = True
    param_random_walks_file_file = "walks/simple-walks.txt"
    param_model_file = "simple_c2b2r_model.npy"
    param_gensim_model_file = "simple_c2b2r_model_gensim.npy"
    param_nodes_file = "data/nodes.csv"
    param_evaluation_set_file = "evaluation/evaluation_set_file"
    param_dataset = Chem2Bio2RDF(param_nodes_file)
    param_walks = MyCorpus(param_random_walks_file_file)
    model = Metapath2Vec(param_walks, param_dataset, param_window_size, param_embedding_dimension,
                         param_number_of_epochs, param_learning_rate,
                         param_min_learning_rate,
                         param_negative_sampling_window_size, param_sample_rate, param_sample_threshold,
                         param_factor_of_negative_sample,
                         param_workers, param_batch_size, param_metapath2vec, param_pp, param_gensim,
                         param_model_file, param_gensim_model_file)
    model.train()
    # Evaluate metapath2vec
    evaluation = Chem2Bio2RDF2Evaluation(param_hits, param_evaluation_set_file, param_model_file,
                                         model)
    evaluation.evaluate()
    # Evaluate gensim
    evaluation_gensim = Chem2Bio2RDF2Evaluation(param_hits, param_evaluation_set_file, param_gensim_model_file,
                                                model)
    evaluation_gensim.evaluate()
