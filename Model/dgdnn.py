import torch
import torch.nn as nn
import torch.nn.functional as F

class DGDNN(nn.Module):
    def __init__(self, relation_size, layer_size, node_feature_size, readout_size, layers, num_nodes, expansion_step):
        """
        Initialize the Decoupled Graph Diffusion Neural Network (DGDNN) model.

        Args:
            relation_size (list): Sizes of relation embeddings at each layer.
            layer_size (list): Sizes of hidden layers for information propagation.
            node_feature_size (list): Sizes of hidden layers for node feature transformation.
            readout_size (list): Sizes of hidden layers for the readout process.
            layers (int): Number of layers.
            num_nodes (int): Number of nodes in the graph.
            expansion_step (int): Number of expansion steps.
        """
        super(DGDNN, self).__init__()

        # Initialize transition matrices and weight coefficients for all layers
        self.T = nn.Parameter(torch.randn(layers, num_nodes, num_nodes))
        self.theta = nn.Parameter(torch.randn(layers, expansion_step))

        # Initialize different module layers at all levels
        self.diffusion_layers = nn.ModuleList(
            [nn.Linear(relation_size[i], relation_size[i + 1]) for i in range(len(relation_size) - 1)])
        self.model_layers = nn.ModuleList(
            [nn.Linear(layer_size[i], layer_size[i + 1]) for i in range(len(layer_size) - 1)])
        self.node_feature_layers = nn.ModuleList(
            [nn.Linear(node_feature_size[i], node_feature_size[i + 1]) for i in range(len(node_feature_size) - 1)])
        self.readout = nn.ModuleList(
            [nn.Linear(readout_size[i], readout_size[i + 1]) for i in range(len(readout_size) - 1)])

        # Initialize activations
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.ReLU()

    def forward(self, X, A):
        """
        Forward pass of the DGDNN model.

        Args:
            X (torch.Tensor): Node feature matrix.
            A (torch.Tensor): Adjacency matrix.

        Returns:
            torch.Tensor: Predicted graph representation.
        """
        # Initialize latent representation with node feature matrix
        z = X

        for q in range(self.T.shape[0]):

            # Select corresponding layer at each level
            diffusion_layers = self.diffusion_layers[q]
            model_layers = self.model_layers[q]
            node_feature_layers = self.node_feature_layers[q]
            theta = self.theta[q]

            z_sum = torch.zeros_like(z)

            # Information diffusion process on graphs
            for i in range(self.T.shape[1]):
                z_sum += (theta[i] * self.T[q][i] * A) @ z

            # Information propagation transform
            z = self.activation1(diffusion_layers(z_sum))

            # Copy current output for next layer
            box = z

            # Node feature transform
            if q != 0:
                f = model_layers(torch.cat((f, node_feature_layers(box)), dim=1))
                f = self.activation2(f)
            else:
                f = model_layers(node_feature_layers(box))
                f = self.activation2(f)

        # Readout process to generate final graph representation
        for readouts in self.readout:
            f = readouts(f)
            if readouts is not self.readout[-1]:
                f = self.activation2(f)

        return F.softmax(f, dim=1)

    def reset_parameters(self):
        """
        Reset model parameters with appropriate initialization methods.
        """
        nn.init.normal_(self.T)
        nn.init.normal_(self.theta)

        for layer in self.diffusion_layers:
            nn.init.kaiming_uniform_(layer.weight)
        for layer in self.model_layers:
            nn.init.kaiming_uniform_(layer.weight)
        for layer in self.node_feature_layers:
            nn.init.kaiming_uniform_(layer.weight)
        for layer in self.readout:
            if layer is self.readout[-1]:
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_uniform_(layer.weight)

 



