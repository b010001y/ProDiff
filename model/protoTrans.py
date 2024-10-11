import torch
import torch.nn as nn
import torch.nn.functional as F

# class TrajectoryTransformer(nn.Module):
#     def __init__(self, input_dim, embed_dim, num_layers, num_heads, forward_dim, seq_len, dropout=0.1):
#         super(TrajectoryTransformer, self).__init__()

#         self.input_dim = input_dim
#         self.embed_dim = embed_dim
#         self.linear_projection = nn.Linear(input_dim, embed_dim)

#         ## 加载预训练模型
#         self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=forward_dim, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         x = self.linear_projection(x)
#         x = x + self.pos_embedding[:, :seq_len, :]
#         x = x.permute(1, 0, 2)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)
#         x = x.sum(dim=1)
#         return x

class TrajectoryTransformer(nn.Module):
    def __init__(self, batch_size, input_dim, embed_dim, num_layers, num_heads, forward_dim, seq_len, dropout=0.1):
        super(TrajectoryTransformer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.linear_projection = nn.Linear(input_dim, embed_dim)

        ## 加载预训练模型
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=forward_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        ## 添加 Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm_features = nn.LayerNorm(embed_dim)
        ## 添加prototypes输出层
        self.output_layer = nn.Linear(batch_size * seq_len, 20)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.linear_projection(x) # (batch_size, seq_len, embed_dim)
        x = self.layer_norm1(x)  # 添加 Layer Normalization
        x = x + self.pos_embedding[:, :seq_len, :]
        x = x.permute(1, 0, 2) # (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        x = self.layer_norm2(x)  # 添加 Layer Normalization
        x = x.permute(1, 0, 2) # (batch_size, seq_len, embed_dim)
        
        features = x.sum(dim=1) # (batch_size, embed_dim)
        features = self.layer_norm_features(features)
        
        x = x.reshape(-1, self.embed_dim)
        x = x.permute(1, 0)
        x = self.output_layer(x) #(embed_dim, 20)
        x = x.permute(1, 0) #(20, embed_dim)
        return x, features
