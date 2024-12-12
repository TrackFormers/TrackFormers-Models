import torch
import pandas as pd
import argparse

from model import TransformerRegressor
from training import clustering
from data_processing.dataset import HitsDataset, get_dataloaders, PAD_TOKEN
from evaluation.plotting import plot_heatmap
from evaluation.scoring import calc_score_trackml
from refiner.training_refiner import refine
from refiner.refining_clusters_dataset import load_data_for_refiner

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_with_refined_clusters(regressor, test_loader, refiner, min_cl_size, min_samples):
    '''
    Evaluates the network regressor on the test data. Returns the predictions and scores.
    After the clustering stage, the refiner is used to increase the purity of the clusters.
    The TrackML score is only calculated after the refiner has run.
    '''
    # Get the regressor in evaluation mode
    torch.set_grad_enabled(False)
    regressor.eval()
    predictions = {}
    score, perfects, doubles, lhcs = 0., 0., 0., 0.

    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        # Use regressor to make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = regressor(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)
        hits = torch.unsqueeze(hits[~padding_mask], 0)

        # Cluster track parameters
        cluster_labels = clustering(pred, min_cl_size, min_samples)

        # Create dataframe from cluster data to provide the refiner with
        rows = []
        for i in range(len(hits[0])):
            rows.append([hits[0][i][0].item(), hits[0][i][1].item(), hits[0][i][2].item(), track_params[0][i][0].item(), track_params[0][i][1].item(), track_params[0][i][2].item(),
                         track_params[0][i][3].item(), cluster_labels[0][i].item(), track_labels[0][i][0].item(), track_labels[0][i][1].item()])
        df = pd.DataFrame(rows, columns=['x','y','z','theta','sinphi','cosphi','q','cluster_id','track_id','weight'])
        
        # Use refiner to refine the clusters
        refiner_pred = refine(refiner, df)
        flattened_cluster_labels = [x for (xs,_) in refiner_pred for x in xs]
        flattened_track_ids = [x for (_,xs) in refiner_pred for x in xs]

        # Calculate scores based on refined clusters
        event_score, scores = calc_score_trackml(flattened_cluster_labels, flattened_track_ids)
        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, flattened_cluster_labels, flattened_track_ids, event_score)

    return predictions, score/len(test_loader), perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--refiner_model_name', type=str)

    parser.add_argument('--nr_enc_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--nr_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    transformer = TransformerRegressor(num_encoder_layers=args.nr_enc_layers,
                                        d_model=args.embedding_size,
                                        n_head=args.nr_heads,
                                        input_size=3,
                                        output_size=4,
                                        dim_feedforward=args.hidden_dim,
                                        dropout=args.dropout)
    transformer = transformer.to(DEVICE)
    checkpoint = torch.load(args.model_name, map_location=torch.device('cpu'))
    transformer.load_state_dict(checkpoint['model_state_dict'])
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    refiner = TransformerRegressor(num_encoder_layers=args.nr_enc_layers,
                                        d_model=args.embedding_size,
                                        n_head=args.nr_heads,
                                        input_size=3,
                                        output_size=4,
                                        dim_feedforward=args.hidden_dim,
                                        dropout=args.dropout)
    refiner = refiner.to(DEVICE)
    checkpoint = torch.load(args.refiner_model_name, map_location=torch.device('cpu'))
    refiner.load_state_dict(checkpoint['model_state_dict'])
    pytorch_total_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    hits_data, track_params_data, track_classes_data = load_data_for_refiner(data=args.data_path, normalize=True)
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                                train_frac=0.7,
                                                                valid_frac=0.15,
                                                                test_frac=0.15,
                                                                batch_size=1)
    print('Data loaded')


    preds, score, perfect, double, lhc = predict_with_refined_clusters(transformer, test_loader, refiner, 5, 2)
    print(score, perfect, double, lhc)
    # preds = list(preds.values())
    # for param in ["theta", "phi", "q"]:
    #     plot_heatmap(preds, param, "test")