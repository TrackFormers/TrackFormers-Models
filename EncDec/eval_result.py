import argparse
from utils import decode_track, Euclid_distance_metric, Normalizer, convert_cylindrical_to_cartesian, generate_data
from models import create_model
import numpy as np
import copy
import pandas as pd

def main(args):
    """
    Evaluates the Transformer model's performance on the dataset.

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
    embed_dim = 64
    dense_dim = 512
    num_heads = 4
    num_induce = args.max_size_hits

    # Build and load the trained model
    transformer = create_model(
        embed_dim, dense_dim, num_heads, num_induce, 
        num_stacks=1, 
        max_size_hits=args.max_size_hits, 
        batch_size=args.batch_size
    )
    transformer.load_weights(args.load_dir)

    # Define distance metric
    dist = Euclid_distance_metric(norm)
    transformer.summary()
    transformer.compile(
        "Adam", loss="MSE", metrics=[dist]
    )

    event_scores = []
    i = 0
    cumsum = 0
    cumlen = 0

    # Iterate over tracks and hits for evaluation
    for track_to_decode, hits_to_decode in zip(tracks, hits):
        # Stop if end of event is reached
        if (sum(sum(abs(track_to_decode))) == 0.5):
            break 

        # Decode the track using the transformer model
        decoded_track = decode_track(
            np.expand_dims(hits_to_decode, axis=0), 
            np.expand_dims(track_to_decode, axis=0), 
            transformer, 
            norm
        )[0]

        print(decoded_track.numpy())
        print(track_to_decode)

        # Find stopping index based on a specific condition
        end_of_track_index = np.where(
            (track_to_decode == (0.1, 0., 0.5)).all(axis=1)
        )[0][0]

        # Compare decoded track with true track
        hit_classification = np.isclose(
            track_to_decode[4:end_of_track_index], 
            decoded_track.numpy()[4:end_of_track_index]
        ).all(axis=1)

        cumsum += sum(hit_classification)
        cumlen += len(hit_classification)

        i += 1

        # Print evaluation metrics
        print(i)
        print(cumsum/cumlen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model results')
    parser.add_argument('--number_of_tracks', type=int, default=1000, help='number of tracks')
    parser.add_argument('--max_size_tracks', type=int, default=30, help='maximum size of tracks')
    parser.add_argument('--max_size_hits', type=int, default=450, help='maximum size of hits')
    parser.add_argument('--file_name', type=str, default='data.csv', help='file name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--load_dir', type=str, default='saved_models/training_temp', help='load directory')
    parser.add_argument('--plotted_track', type=int, default=0, help='plotted track index')
    parser.add_argument('--parsed_data_dir', type=str, default=None, help='Parsed numpy arrays tracks and hits data dir')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory for evaluation results')
    args = parser.parse_args()

    main(args)