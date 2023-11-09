import torch
import torch.nn as nn
import torch.nn.functional as F

class DGDNN(nn.Module):
    def __init__(self, diffusion_size, linear_size, decoupled_size, mlp_size, layers, num_nodes, expansion_step):
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
        self.T = nn.Parameter(torch.randn(layers, expansion_step, num_nodes, num_nodes))
        self.theta = nn.Parameter(torch.randn(layers, expansion_step))

        # Initialize different module layers at all levels
        self.diffusion_layers = nn.ModuleList(
            [GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1]) for i in range(len(diffusion_size) - 1)])
        self.linear_layers = nn.ModuleList(
            [nn.Linear(linear_size[2 * i], linear_size[2 * i + 1]) for i in range(len(linear_size) // 2)])
        self.decoupled_layers = nn.ModuleList(
            [nn.Linear(decoupled_size[2 * i], decoupled_size[2 * i + 1]) for i in range(len(decoupled_size) // 2)])
        self.mlp = nn.ModuleList(
            [nn.Linear(mlp_size[i], mlp_size[i + 1]) for i in range(len(mlp_size) - 1)])

        # Initialize activations
        self.activation2 = nn.LeakyReLU()

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
        h = X

        for q in range(self.T.shape[0]):
          z = self.diffusion_layers[q](self.theta[q], self.T[q], z, A)
          h = self.activation2(self.decoupled_layers[q](torch.cat((self.linear_layers[q](z), h), dim=1)))

        # Readout process to generate final graph representation
        for l in self.mlp:
          h = l(h)
          if l is not self.mlp[-1]:
            h = self.activation2(h)

        return h

    def reset_parameters(self):
        """
        Reset model parameters with appropriate initialization methods.
        """
        nn.init.normal_(self.T)
        nn.init.normal_(self.theta)
 



