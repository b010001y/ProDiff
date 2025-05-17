import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryTransformer(nn.Module):
    """Transformer model to learn trajectory embeddings and a set of learnable prototypes."""
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, forward_dim, seq_len, n_cluster, dropout=0.1):
        """Initializes the TrajectoryTransformer.

        Args:
            input_dim (int): Dimension of the input trajectory points (e.g., 3 for time, lat, lon).
            embed_dim (int): Dimension of the embeddings within the transformer.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in the transformer.
            forward_dim (int): Dimension of the feed-forward network in transformer layers.
            seq_len (int): Length of the input trajectory sequences.
            n_cluster (int): Number of prototypes to learn.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(TrajectoryTransformer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_cluster = n_cluster

        self.linear_projection = nn.Linear(input_dim, embed_dim)

        # Positional embedding for the sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=forward_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer Normalization layers
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim) 
        self.layer_norm_features = nn.LayerNorm(embed_dim)
        
        # Learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(n_cluster, embed_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights of the linear layers, transformer components, and prototypes."""
        nn.init.xavier_uniform_(self.linear_projection.weight)
        if self.linear_projection.bias is not None:
            nn.init.zeros_(self.linear_projection.bias)
        
        # Initialize transformer layers (already done by PyTorch's default, but can be overridden)
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.linear1.weight)
            if layer.linear1.bias is not None: nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight)
            if layer.linear2.bias is not None: nn.init.zeros_(layer.linear2.bias)
            # Self-attention weights are more complex (in_proj_weight, out_proj.weight)
            # Default Pytorch init is usually fine for these.
            if hasattr(layer.self_attn, 'in_proj_weight') and layer.self_attn.in_proj_weight is not None:
                 nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if hasattr(layer.self_attn, 'in_proj_bias') and layer.self_attn.in_proj_bias is not None:
                 nn.init.zeros_(layer.self_attn.in_proj_bias)
            if hasattr(layer.self_attn.out_proj, 'weight') and layer.self_attn.out_proj.weight is not None:
                 nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            if hasattr(layer.self_attn.out_proj, 'bias') and layer.self_attn.out_proj.bias is not None:
                 nn.init.zeros_(layer.self_attn.out_proj.bias)

        # Initialize prototypes (e.g., Xavier uniform)
        nn.init.xavier_uniform_(self.prototypes.data)

    def forward(self, x):
        """Forward pass of the TrajectoryTransformer.

        Args:
            x (torch.Tensor): Input trajectory batch, shape (batch_size, seq_len, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - prototypes (torch.Tensor): Learned prototypes, shape (n_cluster, embed_dim).
                - features (torch.Tensor): Trajectory features, shape (batch_size, embed_dim).
        """
        batch_size, seq_len, _ = x.size()
        
        x = self.linear_projection(x)  # Project to (batch_size, seq_len, embed_dim)
        x = self.layer_norm1(x)      # Apply layer normalization
        x = x + self.pos_embedding[:, :seq_len, :] # Add positional embedding
        
        x = self.transformer_encoder(x) # Input: (batch_size, seq_len, embed_dim)
        
        x = self.layer_norm2(x)  # Apply layer normalization after transformer
        
        # Aggregate features from the sequence (e.g., by summing along sequence length)
        features = x.sum(dim=1)  # (batch_size, embed_dim)
        features = self.layer_norm_features(features) # Normalize aggregated features
        
        # The first returned value `prototypes_from_transformer` in train.py was from the old `output_layer`.
        # Now we return the learnable self.prototypes.
        return self.prototypes, features

