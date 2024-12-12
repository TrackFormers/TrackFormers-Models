import argparse
from utils import preprocess, decode_track, Euclid_distance_metric, Normalizer, convert_cylindrical_to_cartesian
from models import create_model
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt

def main(args):
    """
    Generates and saves a 3D plot of the decoded and true tracks.

    Args:
        args (Namespace): Command-line arguments containing configuration parameters.
    """
    # Print input parameters
    print("Number of tracks: ", args.number_of_tracks)
    print("Maximum size of tracks: ", args.max_size_tracks)
    print("Maximum size of hits: ", args.max_size_hits)
    print("File name: ", args.file_name)
    print("Batch size: ", args.batch_size)
    print("Load directory: ", args.load_dir)
    print("Plotted track: ", args.plotted_track)
    print("Parsed data directory: ", args.parsed_data_dir)
    print("Output directory: ", args.output_dir)

    # Load data from CSV
    df = pd.read_csv(args.file_name, delimiter=';')

    # Initialize normalizer with data ranges
    norm = Normalizer(
        df["hit_r"].min(), df["hit_r"].max(),
        df["hit_theta"].min(), df["hit_theta"].max(),
        df["hit_z"].min(), df["hit_z"].max()
    )

    # Load preprocessed tracks and hits
    tracks = np.load(os.path.join(args.parsed_data_dir, "tracks.npy"))
    hits = np.load(os.path.join(args.parsed_data_dir, "hits.npy"))

    # Define model hyperparameters
    embed_dim = 128
    dense_dim = 2048
    num_heads = 16
    num_induce = args.max_size_hits

    # Build and load the trained model
    transformer = create_model(
        embed_dim, dense_dim, num_heads, num_induce, 
        max_size_hits=args.max_size_hits, batch_size=args.batch_size
    )
    transformer.load_weights(args.load_dir)

    # Define distance metric
    dist = Euclid_distance_metric(norm)
    transformer.summary()
    transformer.compile(
        "Adam", loss="MSE", metrics=[dist]
    )

    # Select the track to plot
    i = args.plotted_track

    # Deep copy data to avoid modifying original arrays
    hits_to_decode = copy.deepcopy(hits)
    tracks_to_decode = copy.deepcopy(tracks)

    # Decode the selected track
    decoded_track = decode_track(
        hits_to_decode[i:i+1, :], 
        tracks_to_decode[i:i+1, :-1], 
        transformer, 
        norm
    )[0]
    true_track = tracks_to_decode[i:i+1, :-1][0]

    decoded_track = decoded_track.numpy()

    # Determine stopping index based on specific condition
    stop_index = np.where(
        (decoded_track.round(2)[:,0] == 0.1) & 
        (decoded_track.round(2)[:,1] == 0.) & 
        (decoded_track.round(2)[:,2] == 0.5)
    )[0]
    if stop_index.size == 0:
        stop_index = decoded_track.shape[0]
    else:
        stop_index = stop_index[0]
    decoded_track_stopped = decoded_track[:stop_index]

    # Extract true hits, removing zero entries
    true_hits = hits_to_decode[i, :]
    true_hits = true_hits[~np.all(true_hits == 0, axis=1)]

    # Unnormalize the decoded and true tracks and hits
    decoded_track_stopped[:, 0], decoded_track_stopped[:, 1], decoded_track_stopped[:, 2] = \
        norm.unnormalize(decoded_track_stopped[:, 0], decoded_track_stopped[:, 1], decoded_track_stopped[:, 2])
    true_track[:, 0], true_track[:, 1], true_track[:, 2] = \
        norm.unnormalize(true_track[:, 0], true_track[:, 1], true_track[:, 2])
    true_hits[:, 0], true_hits[:, 1], true_hits[:, 2] = \
        norm.unnormalize(true_hits[:, 0], true_hits[:, 1], true_hits[:, 2])

    # Convert cylindrical coordinates to Cartesian for plotting
    x_vec_reco, y_vec_reco, z_vec_reco = convert_cylindrical_to_cartesian(
        decoded_track_stopped[:, 0], decoded_track_stopped[:, 1], decoded_track_stopped[:, 2]
    )
    x_vec_true, y_vec_true, z_vec_true = convert_cylindrical_to_cartesian(
        true_track[:, 0], true_track[:, 1], true_track[:, 2]
    )
    x_vec_true_hit, y_vec_true_hit, z_vec_true_hit = convert_cylindrical_to_cartesian(
        true_hits[:, 0], true_hits[:, 1], true_hits[:, 2]
    )

    # Initialize 3D plot
    ax = plt.axes(projection='3d')

    # Plot true hits with transparency
    ax.scatter3D(x_vec_true_hit, y_vec_true_hit, z_vec_true_hit, cmap='viridis', alpha=0.4)

    # Plot true track segments
    ax.plot3D(x_vec_true[1:10], y_vec_true[1:10], z_vec_true[1:10], 'green')
    ax.scatter3D(x_vec_true[1:10], y_vec_true[1:10], z_vec_true[1:10], cmap='viridis')

    # Plot reconstructed track segments
    ax.plot3D(x_vec_reco[1:10], y_vec_reco[1:10], z_vec_reco[1:10], 'red')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the plot to the specified output directory
    plt.savefig(os.path.join(args.output_dir, f"track_{i}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot model results')
    parser.add_argument('--number_of_tracks', type=int, default=1000, help='number of tracks')
    parser.add_argument('--max_size_tracks', type=int, default=30, help='maximum size of tracks')
    parser.add_argument('--max_size_hits', type=int, default=450, help='maximum size of hits')
    parser.add_argument('--file_name', type=str, default='data.csv', help='file name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--load_dir', type=str, default='saved_models/training_temp', help='load directory')
    parser.add_argument('--plotted_track', type=int, default=0, help='plotted track index')
    parser.add_argument('--parsed_data_dir', type=str, default=None, help='Parsed numpy arrays tracks and hits data dir')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory for plots')
    args = parser.parse_args()

    main(args)