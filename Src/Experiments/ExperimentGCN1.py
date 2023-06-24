import tensorflow as tf
from Src.DataPreparation.CoraCitationNetwork import CoraCitationNetwork
from Src.DataPreparation.KarateClubNetwork import KarateClubNetwork
from Src.Graph.DeepLearningModels.GCNModel import GCNModel
from Src.Graph.Utils import AlgorithmicUtils


@AlgorithmicUtils.measure_execution_time
def main():
    # Start Tensorflow
    print("Main Program")
    print("Tensorflow version: " + tf.__version__)
    if tf.test.is_built_with_cuda():
        print("Tensorflow built with CUDA support")
    else:
        print("Tensorflow is NOT built with CUDA support")
    print(tf.config.list_physical_devices("CPU"))
    print(tf.config.list_physical_devices("GPU"))
    home_directory = "G:/My Drive/Work/Research/TWU/Projects/Network Science Toolkit/Code/NetworkScienceToolkit/"
    # Hyper Parameters
    pararmeters = AlgorithmicUtils.read_config(home_directory + "Config/config.txt")
    dataset = str(pararmeters["dataset"])
    if dataset == "karate":
        # Load the Karate Club Network-----------------------------------------------------------
        data = KarateClubNetwork(home_directory + "Data/Karate/Raw/karate.txt",
                                 home_directory + "Data/Karate/Raw/karate-node-labels.txt")
        data.load()
        data.graph.info()
        print(data.graph.schema.keys())
        print(data.graph.node_type_profile)
        embedding_file = home_directory + "Data/Karate/Embedding/karate_embedding.tsv"
        metadata_file = home_directory + "Data/Karate/Embedding/karate_metadata.tsv"
    elif dataset == "cora":
        # Load the Cora Network -----------------------------------------------------------
        data = CoraCitationNetwork(home_directory + "Data/Cora/Raw/nodes.csv",
                                   home_directory + "Data/Cora/Raw/edges.csv")
        data.load()
        data.graph.info()
        print(data.graph.schema.keys())
        print(data.graph.node_type_profile)
        embedding_file = home_directory + "Data/Cora/Embedding/cora_embedding.tsv"
        metadata_file = home_directory + "Data/Cora/Embedding/cora_metadata.tsv"
    model_nodes = GCNModel(data.graph, pararmeters)
    model_nodes.train_model()
    model_nodes.test_model()
    model_nodes.save_model(embedding_file, metadata_file)


if __name__ == '__main__':
    main()
