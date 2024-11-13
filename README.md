# Graph-Based Clustering and Link Prediction Model

This repository implements a model for graph-based clustering and link prediction, leveraging PyTorch Geometric, graph neural networks, and text embeddings for enhanced node feature representation. The code facilitates the encoding and processing of graph data, model training, and performance evaluation on various metrics.

## Requirements

This project requires the following packages:

- PyTorch
- PyTorch Geometric
- Transformers
- Scikit-Learn
- NetworkX
- Matplotlib
- Pandas
- NumPy
- Scipy

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-geometric transformers scikit-learn networkx matplotlib pandas numpy scipy




## Code Structure and Explanation

### Key Classes and Functions

- **dataset_vibgnn**: A custom dataset class for representing graph data, including edge lists, node features, and classes. It supports:
  - **Graph construction** with node features and edge weights.
  - **Negative example generation** for link prediction tasks, where negative edges are created by randomly selecting node pairs that are not connected.
  - **Training graph creation** with optional hidden edges, allowing for testing on withheld data.

- **build_adjacency_matrix**: Constructs a sparse adjacency matrix from an edge list. This matrix is primarily used in graph processing tasks for various matrix-based computations, ensuring all nodes are connected appropriately within the graph.

- **conf_matrix**: A function to compute the true positive rate (TPR) from a confusion matrix, which is used for evaluating the accuracy of classification.

- **eval**: Evaluates the model by computing the ROC AUC score on test data, measuring how well the model can distinguish between classes.

- **GAT, GCN, and MLP**: Neural network modules implementing different architectures for node feature learning:
  - **GAT**: Graph Attention Network (GAT) layers for learning node representations with attention mechanisms, focusing on the most relevant neighboring nodes.
  - **GCN**: Graph Convolutional Network (GCN) layers for learning node representations by aggregating neighboring node features.
  - **MLP**: A Multi-Layer Perceptron (MLP) for transforming and projecting feature vectors into a desired output space.

- **VIB**: A model class that combines GNN layers and distance computation for link prediction. It includes:
  - `distance`: Computes the distance between node pairs.
  - `forward`: Performs forward propagation, calculating probabilities based on distances.
  - `loss`: Calculates the loss for training, including classification and clustering loss components.

- **visualize**: A function for visualizing node embeddings using t-SNE for dimensionality reduction.

- **encode_and_build**: Preprocesses text data, builds node embeddings, and constructs the final graph dataset. It includes steps for:
  - Encoding text features using pre-trained language models.
  - Building the graph and generating training/testing datasets.

- **run_model**: The main function for running the model, including data loading, graph construction, model training, and evaluation. It accepts parameters for model type, data paths, clustering type, and text embeddings.


## How to Run the Code

To run the code, follow these steps:

1. **Set Up Data Files**: Ensure you have the following data files:
   - `*_mapping.csv`: Contains node IDs and raw text features (tweets).
   - `*_graph.txt`: Contains graph edge list information.

2. **Run the Model**: Customize parameters and execute `run_model`. The key parameters include:
   - `map_path`: Path to the mapping CSV file.
   - `edge_path`: Path to the edge list text file.
   - `inv`: Boolean to specify inverted adjacency matrix usage.
   - `hom`: Boolean to specify homophilic or heterophilic graph version.
   - `use_txt`: Boolean to specify the use of text embeddings.
   - `lr`: Learning rate for optimization.
   - `X`: Optional feature matrix for nodes.
   - `lm`: Language model for text embeddings (e.g., `distilbert-base-uncased`).
   - `no_of_clusters`: Number of clusters for node classification.

Example:
```python
# Running the model on 'cd' dataset with clustering
emb, acc_rec_final, acc_final, f_final, in_final, X = run_model(
    map_path='cd_mapping.csv',
    edge_path='cd_graph.txt',
    inv=False,
    hom=False,
    use_txt=True,
    lr=0.0001,
    X=None,
    lm='distilbert-base-uncased',
    no_of_clusters=2
)
3. **Evaluate Performance**: After training, performance metrics such as accuracy and F1-score are printed. These metrics are averaged over multiple runs for stability.

## Example Datasets

Sample datasets include:
- `cd`: Contains topics such as `abortion`, `marijuana`, and `gayRights`.
- `timme`: TIMME dataset with either `TIMME-All` or `TIMME-Pure` variants.

Configure dataset selection and topics in the script:
```python
dat = 'cd'
top = 'abortion'  # specify topic
t_all = True  # use TIMME-All
path = ''  # specify data directory path


## Citation

If you use this code, please cite the respective papers and sources for the models and datasets used.
