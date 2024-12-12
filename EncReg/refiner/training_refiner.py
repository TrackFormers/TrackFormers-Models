import torch
import torch.nn as nn
import numpy as np

from model import TransformerRegressor, save_model
from data_processing.dataset import get_dataloaders, PAD_TOKEN
from refiner.refining_clusters_dataset import load_data_for_refiner, ClustersDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, optim, train_loader, loss_fn):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.

    for data in train_loader:
        _, hits, labels = data
        optim.zero_grad()

        # Make prediction
        hits = hits.to(DEVICE)
        labels = labels.to(DEVICE)
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = pred[~padding_mask]
        labels = labels[~padding_mask]

        # Calculate loss and use it to update weights
        loss = loss_fn(pred, labels)
        final_loss = loss
        final_loss.backward()
        optim.step()
        losses += final_loss.item()

    return losses / len(train_loader)


def evaluate(model, validation_loader, loss_fn):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    model.eval()
    losses = 0.
    with torch.no_grad():
        for data in validation_loader:
            _, hits, labels = data

            # Make prediction
            hits = hits.to(DEVICE)
            labels = labels.to(DEVICE)
            
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            pred = model(hits, padding_mask)

            pred = pred[~padding_mask]
            labels = labels[~padding_mask]

            # Calculate loss and use it to update weights
            loss = loss_fn(pred, labels)
            losses += loss.item()
            
    return losses / len(validation_loader)


def refine(model, data):
    '''
    Use model to refine the predicted clusters contained in the dataframe data.
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()

    refined_clusters = []
    loss_fn = nn.MSELoss(reduction='none')
    # For every predicted cluster by the Transformer Regressor
    for cl_id in data['cluster_id'].unique().tolist():
        rows = data[data['cluster_id'] == cl_id]
        hits = torch.tensor(rows[['x','y','z']].to_numpy(dtype=np.float32))
        cluster_ids = torch.tensor(rows['cluster_id'].to_numpy(dtype=int))
        track_ids = torch.tensor(rows[['track_id','weight']].to_numpy(dtype=np.float32))

        # Pass the cluster of hits to the refiner network
        padding_mask = (hits == PAD_TOKEN).all(dim=1)
        pred = model(hits, padding_mask)
        representative_track_params, _ = torch.median(pred, dim=0)
        representative_track_params = representative_track_params.unsqueeze(0).repeat(len(rows), 1)

        # Calculate the loss between the median prediction and every track parameter set
        losses = loss_fn(pred, representative_track_params).mean(dim=1)
        cutoff = 0.005
        # Remove the hits with MSE > cutoff
        acceptable_hits_in_cluster = cluster_ids[losses < (min(losses)+cutoff)]
        coresponding_track_ids = track_ids[losses < (min(losses)+cutoff)]
        # Keep the changes only if it doesn't make the cluster smaller than 4 hits
        if len(acceptable_hits_in_cluster) < 4:
            refined_clusters.append((cluster_ids, track_ids))
        else:
            refined_clusters.append((acceptable_hits_in_cluster, coresponding_track_ids))

    return refined_clusters


if __name__ == "__main__":
    NUM_EPOCHS = 100
    EARLY_STOPPING = 50
    MODEL_NAME = "refiner"

    torch.manual_seed(37)  # for reproducibility

    # Load dataset into dataloader and split into train and validation set
    hits_data, labels_data = load_data_for_refiner(data_path="trackml_10to50tracks_40kevents.csv", normalize=True)

    dataset = ClustersDataset(hits_data, labels_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                            train_frac=0.7,
                                            valid_frac=0.15,
                                            test_frac=0.15,
                                            batch_size=32)
    print("data loaded")

    # Transformer refiner model
    transformer = TransformerRegressor(num_encoder_layers=3,
                                        d_model=32,
                                        n_head=2,
                                        input_size=3,
                                        output_size=4,
                                        dim_feedforward=64,
                                        dropout=0.1)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    loss_fn = nn.MSELoss() 
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn)

        # Evaluate using validation split
        val_loss = evaluate(transformer, valid_loader, loss_fn)

        # Bookkeeping
        print(f"Epoch: {epoch}\nVal loss: {val_loss:.8f}, Train loss: {train_loss:.8f}", flush=True)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count, MODEL_NAME)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count, MODEL_NAME)
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break