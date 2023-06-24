from Src.Graph.DeepLearningModels.TransEModel import TransEModel
import tensorflow as tf
def main():
    data_file="G:/My Drive/Library/Data/WordNet/wordnet-mlj12/wordnet-mlj12-train.txt"
    data=[]
    entities=[]
    relations=[]
    with open(data_file, 'r') as f:
        for line in f:
            triple=line.split("\t")
            h=triple[0].strip()
            l=triple[1].strip()
            t=triple[2].strip()
            data.append((h,l,t))
            entities.append(h)
            entities.append(t)
            relations.append(l)
    entities=list(set(entities))
    relations = list(set(relations))
    # train_dataset = tf.data.Dataset.from_tensor_slices(data)
    # batches = train_dataset.batch(100)
    # for batch in batches:
    #     for triple in batch:
    #         h=str(triple.numpy()[0].decode('utf-8'))
    #         l=str(triple.numpy()[1].decode('utf-8'))
    #         t=str(triple.numpy()[2].decode('utf-8'))
    #         print(h,l,t)
    te=TransEModel(data,entities,relations,2,128,100,1000,0.01)
    te.train()



if __name__ == "__main__":
    main()