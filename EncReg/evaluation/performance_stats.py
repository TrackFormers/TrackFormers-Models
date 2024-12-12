import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np
import argparse
from time import process_time_ns

from training import clustering
from model import TransformerRegressor
from data_processing.dataset import HitsDataset, get_dataloaders, PAD_TOKEN, load_curved_3d_data, load_linear_3d_data, load_linear_2d_data
from data_processing.trackml_data import load_trackml_data
from evaluation.plotting import plot_heatmap
from evaluation.scoring import calc_score, calc_score_trackml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_with_stats(model, test_loader, min_cl_size, min_samples, data_type):
    '''
    Evaluates model on test_loader, using the specified HDBSCAN parameters,
    and returns the average MSE, TrackML score, perfect match efficiency,
    double majority match efficiency, LHC-style match efficiency, and 
    standard deviation of predictions.
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    score, perfects, doubles, lhcs = 0., 0., 0., 0.

    # Transformer physics performance
    loss_fn = nn.MSELoss()
    pred_true_differences = []
    theta_errors, sinphi_errors, cosphi_errors, q_errors = [],[],[],[]

    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        difference = loss_fn(pred, track_params)
        pred_true_differences.append(difference.item())

        cluster_labels = clustering(pred, min_cl_size, min_samples)
        diff = (pred - track_params)[0]
        for line in diff:
            theta_errors.append(line[0].item())
            sinphi_errors.append(line[1].item())
            cosphi_errors.append(line[2].item())
            q_errors.append(line[3].item())

        if data_type == 'trackml':
            event_score, scores = calc_score_trackml(cluster_labels[0], track_labels[0])
        else:
             event_score, scores = calc_score(cluster_labels[0], track_labels[0])
        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)

    print(f'Avg MSE: {sum(pred_true_differences)/len(pred_true_differences)}')
    print(f'Std MSE: {np.std(pred_true_differences)}')
    print(f'TrackML score: {score/len(test_loader)}')
    print(perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader))

    print("Standard deviations of theta, sinphi, cosphi, q:")
    print(np.std(theta_errors))
    print(np.std(sinphi_errors))
    print(np.std(cosphi_errors))
    print(np.std(q_errors))

    return predictions


def measure_speed(model, test_loader, min_cl_size, min_samples):
    '''
    Evaluates model on test_loader, using the specified HDBSCAN parameters,
    and returns the CPU and GPU time of the pipeline.
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()

    # Time performance bookkeeping
    cuda_times = []
    cpu_times = []
    start_event = cuda.Event(enable_timing=True)
    end_event = cuda.Event(enable_timing=True)

    for data in test_loader:
        _, hits, track_params, track_labels = data

        start_event.record()
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        with torch.cuda.amp.autocast():
            pred = model(hits, padding_mask)
        
        end_event.record()
        cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        cuda_times.append(elapsed_time_ms)

        start_cpu_time = process_time_ns()
        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)
        _ = clustering(pred, min_cl_size, min_samples)

        end_cpu_time = process_time_ns()
        cpu_times.append(end_cpu_time - start_cpu_time)

    print("Avg CUDA time:", sum(cuda_times[1:])/len(cuda_times[1:]))
    print("Avg CPU time:", sum(cpu_times[1:])/len(cpu_times[1:]))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_nr_hits', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_type', type=str, choices=['2d', 'linear', 'curved', 'trackml'])

    parser.add_argument('--nr_enc_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--nr_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    data_func = None
    in_size = 3
    out_size = 3
    if args.data_type == '2d':
        data_func = load_linear_2d_data
        in_size = 2
        out_size = 1
        params = ['slope']
    elif args.data_type == 'linear':
        data_func = load_linear_3d_data
        params = ["theta", "sinphi", "cosphi"]
    elif args.data_type == 'curved':
        data_func = load_curved_3d_data
        params = ["radial_coeff", "pitch_coeff", "azimuthal_coeff"]
    elif args.data_type == 'trackml':
        data_func = load_trackml_data
        out_size = 4
        params = ["theta", "sinphi", "cosphi", "q"]

    transformer = TransformerRegressor(num_encoder_layers=args.nr_enc_layers,
                                        d_model=args.embedding_size,
                                        n_head=args.nr_heads,
                                        input_size=in_size,
                                        output_size=out_size,
                                        dim_feedforward=args.hidden_dim,
                                        dropout=args.dropout)
    transformer = transformer.to(DEVICE)

    checkpoint = torch.load(args.model_name, map_location=torch.device('cpu'))
    transformer.load_state_dict(checkpoint['model_state_dict'])
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    hits_data, track_params_data, track_classes_data = data_func(data=args.data_path, max_num_hits=args.max_nr_hits)
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                                train_frac=0.7,
                                                                valid_frac=0.15,
                                                                test_frac=0.15,
                                                                batch_size=1)
    print('data loaded')

    min_cl_size, min_samples = 5, 2
    _ = measure_speed(transformer, test_loader, min_cl_size, min_samples)

    preds = predict_with_stats(transformer, test_loader, min_cl_size, min_samples, args.data_type)

    preds = list(preds.values())
    for param in params:
        plot_heatmap(preds, param, args.model_name)
