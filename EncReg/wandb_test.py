import torch
import wandb

from model import TransformerRegressor
from training_flash import *
from data_processing.dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from data_processing.trackml_data import load_trackml_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(37)

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'test_score'
        },
    'parameters': {
        'num_enc_layers': {'min': 5, 'max': 7},
        'emb_dim': {'values': [128]},
        'nr_heads': {'values': [4, 8]},
        'dim_ff': {'values': [256]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="trackml_200to500tracks")

hits_data, track_params_data, track_classes_data = load_trackml_data(data="../../200to500tracks_40k_old.csv", max_num_hits=5000, normalize=True)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)

def train_model(num_enc_layers, emb_dim, nr_heads, dim_ff):
    transformer = TransformerRegressor(num_encoder_layers=num_enc_layers,
                                                            d_model=emb_dim,
                                                            n_head=nr_heads,
                                                            input_size=3,
                                                            output_size=4,
                                                            dim_feedforward=dim_ff,
                                                            dropout=0.1, use_flashattn=True)
    transformer = transformer.to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    for _ in range(30):
        # Train the model
        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn, scaler)
        val_loss = evaluate(transformer, valid_loader, loss_fn)
        wandb.log({"val_loss" : val_loss, "train_loss" : train_loss})

    _, score, _, _, _ = predict(transformer, test_loader, 5, 2)
    wandb.log({"test_score": score})

def main():
    run = wandb.init()
    num_enc_layers = wandb.config.num_enc_layers
    emb_dim = wandb.config.emb_dim
    nr_heads = wandb.config.nr_heads
    dim_ff = wandb.config.dim_ff
    if emb_dim % nr_heads == 0:
        train_model(num_enc_layers, emb_dim, nr_heads, dim_ff)


wandb.agent(sweep_id, function=main)
wandb.finish()
