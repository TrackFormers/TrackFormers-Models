import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        inputfeature_dim,
        num_classes,
        num_heads,
        embed_dim,
        num_layers,
        dropout=0.0,
    ):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(inputfeature_dim, embed_dim)
        # Default values in nn.TransformerEncoder:
        # dim_feedforward=2048
        # dropout=0.1
        # activation=relu
        # batch_first=false i.e. input&output as (seq,batch,feature)
        # norm_first=false i.e. layer norm after atten.&feedfwd.
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.embedding(x)

        padding_mask = x.sum(dim=-1) == 0

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # x = self.transformer_encoder(x)
        x = self.classifier(x)
        return x
