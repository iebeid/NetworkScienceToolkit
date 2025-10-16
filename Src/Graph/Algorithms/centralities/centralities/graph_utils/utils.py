import numpy as np

class Utils():
    def __init__(self):
        pass

    @staticmethod
    def text_to_dict(filepath):
        entity2class = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.split('\t')
                entity = line[0]
                cls = line[1]
                entity2class[entity] = cls

        return entity2class