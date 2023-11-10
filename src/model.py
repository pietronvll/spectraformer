import torch


class SpectraFormer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        transformer_mlp_dim: int = 512,
        embedding_dim: int = 16,
        num_heads: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = torch.nn.Linear(input_dim, embedding_dim)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.lin_final = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.lin_final(x).squeeze(-1)
        return x
        x = self.lin_final(x).squeeze(-1)
        return x
