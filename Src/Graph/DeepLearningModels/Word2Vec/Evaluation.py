import pandas as pd
import random
import math
from urllib.parse import urlparse

def map_to_letter(number):
    map_value = 0
    if number == '0':
        map_value = 'a'
    if number == '1':
        map_value = 'b'
    if number == '2':
        map_value = 'c'
    if number == '3':
        map_value = 'd'
    if number == '4':
        map_value = 'e'
    if number == '5':
        map_value = 'f'
    if number == '6':
        map_value = 'g'
    if number == '7':
        map_value = 'h'
    if number == '8':
        map_value = 'i'
    if number == '9':
        map_value = 'j'
    return map_value


def map_to_number(number):
    map_value = 0
    if number == 'a':
        map_value = '0'
    if number == 'b':
        map_value = '1'
    if number == 'c':
        map_value = '2'
    if number == 'd':
        map_value = '3'
    if number == 'e':
        map_value = '4'
    if number == 'f':
        map_value = '5'
    if number == 'g':
        map_value = '6'
    if number == 'h':
        map_value = '7'
    if number == 'i':
        map_value = '8'
    if number == 'j':
        map_value = '9'
    return map_value


def map_code_to_letter(code):
    map_value = ''
    for c in code:
        single_map_value = map_to_letter(c)
        map_value = map_value + str(single_map_value)
    return map_value


def map_code_to_number(code):
    map_value = ''
    for c in code:
        single_map_value = map_to_number(c)
        map_value = map_value + str(single_map_value)
    return int(map_value)


def convert_entity_to_code(entity,nodes_dataframe):
    entity_index = nodes_dataframe.index[nodes_dataframe["filtered_node_name"] == str(entity)]
    entity_code = None
    if not entity_index.empty:
        entity_id = nodes_dataframe.at[entity_index[0], "node_id"]
        entity_code = map_code_to_letter(str(entity_id))
    return entity_code

def main():
    positive_file = "data/positive.txt"
    negative_file = "data/negative.txt"
    nodes_file = "data/nodes.csv"
    positive_dataframe = pd.read_csv(positive_file, sep="\t", header=None, encoding='utf-8')
    negative_dataframe = pd.read_csv(negative_file, sep="\t", header=None, encoding='utf-8')
    nodes_dataframe = pd.read_csv(nodes_file, sep=",", encoding='utf-8')
    list_filtered_entites = []
    for index, row in nodes_dataframe.iterrows():
        parsed = urlparse(row[1])
        parts = parsed.path.split("/")
        entity = parts[-1].strip("0")
        filtered_entity = entity.split(":")[-1].strip("0")
        list_filtered_entites.append(filtered_entity)
    nodes_dataframe["filtered_node_name"] = list_filtered_entites
    test_samples = 1000
    sample_tuples = []
    for index in range(0,test_samples):
        random_index_1 = random.randint(0, len(positive_dataframe.index)-1)
        random_index_2 = random.randint(0, len(positive_dataframe.index)-1)
        first_sample = positive_dataframe.loc[random_index_1]
        second_sample = positive_dataframe.loc[random_index_2]
        a = convert_entity_to_code(first_sample[0],nodes_dataframe)
        b = convert_entity_to_code(first_sample[1],nodes_dataframe)
        c = convert_entity_to_code(second_sample[0],nodes_dataframe)
        expected = convert_entity_to_code(second_sample[1],nodes_dataframe)
        sample_tuple = (a,b,c,expected)
        sample_tuples.append(sample_tuple)
    evaluation_samples = pd.DataFrame(sample_tuples)
    evaluation_samples = evaluation_samples.dropna()
    evaluation_samples.to_csv("evaluation-samples.txt",index=False,sep="\t")
    print(evaluation_samples)


if __name__ == "__main__":
    main()