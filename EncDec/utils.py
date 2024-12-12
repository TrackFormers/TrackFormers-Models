import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
import copy
import json

def convert_cylindrical_to_cartesian(r, theta, z):
    """
    Converts cylindrical coordinates (r, theta, z) to Cartesian coordinates (x, y, z).

    Args:
        r (float): Radial distance
        theta (float): Angle in radians
        z (float): Height along the z-axis

    Returns:
        tuple: Cartesian coordinates (x, y, z)
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


class Normalizer:
    """
    A class for normalizing and unnormalizing data across three dimensions.
    """

    def __init__(self, dim_1_min, dim_1_max, dim_2_min, dim_2_max, dim_3_min, dim_3_max):
        """
        Initializes the Normalizer with the min and max values for each dimension.

        Args:
            dim_1_min (float): Minimum value for dimension 1
            dim_1_max (float): Maximum value for dimension 1
            dim_2_min (float): Minimum value for dimension 2
            dim_2_max (float): Maximum value for dimension 2
            dim_3_min (float): Minimum value for dimension 3
            dim_3_max (float): Maximum value for dimension 3
        """
        self.dim_1_min = dim_1_min
        self.dim_1_max = dim_1_max
        self.dim_2_min = dim_2_min
        self.dim_2_max = dim_2_max
        self.dim_3_min = dim_3_min
        self.dim_3_max = dim_3_max

    def normalize(self, dim_1, dim_2, dim_3):
        """
        Normalizes the values of the three dimensions.

        Args:
            dim_1 (array-like): Values for dimension 1
            dim_2 (array-like): Values for dimension 2
            dim_3 (array-like): Values for dimension 3

        Returns:
            tuple: Normalized values for dimensions 1, 2, and 3
        """
        dim_1_norm = (dim_1 - self.dim_1_min) / (self.dim_1_max - self.dim_1_min)
        dim_2_norm = (dim_2 - self.dim_2_min) / (self.dim_2_max - self.dim_2_min)
        dim_3_norm = (dim_3 - self.dim_3_min) / (self.dim_3_max - self.dim_3_min)
        return dim_1_norm, dim_2_norm, dim_3_norm

    def unnormalize(self, dim_1, dim_2, dim_3):
        """
        Reverses the normalization to return to the original scale.

        Args:
            dim_1 (array-like): Normalized values for dimension 1
            dim_2 (array-like): Normalized values for dimension 2
            dim_3 (array-like): Normalized values for dimension 3

        Returns:
            tuple: Original values for dimensions 1, 2, and 3
        """
        dim_1_unnorm = dim_1 * (self.dim_1_max - self.dim_1_min) + self.dim_1_min
        dim_2_unnorm = dim_2 * (self.dim_2_max - self.dim_2_min) + self.dim_2_min
        dim_3_unnorm = dim_3 * (self.dim_3_max - self.dim_3_min) + self.dim_3_min
        return dim_1_unnorm, dim_2_unnorm, dim_3_unnorm
    
class EuclidDistanceMetric:
    """
    A class to calculate the Euclidean distance between true and predicted hits, 
    with optional conversion from cylindrical to Cartesian coordinates.
    """

    def __init__(self, norm, cylindrical=True):
        """
        Initialize the distance metric with a normalizer and cylindrical mode.

        Args:
            norm (Normalizer): Instance of the Normalizer class for coordinate transformation.
            cylindrical (bool): Whether the coordinates are in cylindrical form. Defaults to True.
        """
        self.norm = norm
        self.cylindrical = cylindrical

    @staticmethod
    def convert_cylindrical_to_cartesian(r, theta, z):
        """
        Converts cylindrical coordinates (r, theta, z) to Cartesian (x, y, z).

        Args:
            r (tf.Tensor): Radial distance
            theta (tf.Tensor): Angle in radians
            z (tf.Tensor): Height along the z-axis

        Returns:
            tuple: Cartesian coordinates (x, y, z)
        """
        x = r * tf.math.cos(theta)
        y = r * tf.math.sin(theta)
        return x, y, z

    def __call__(self, y_true, y_pred):
        """
        Calculate the Euclidean distance between true and predicted coordinates.

        Args:
            y_true (tf.Tensor): Ground truth coordinates, shape (batch, hits, 3)
            y_pred (tf.Tensor): Predicted coordinates, shape (batch, hits, 3)

        Returns:
            tf.Tensor: Euclidean distance metric.
        """
        # Unnormalize true and predicted coordinates
        dim_1_true, dim_2_true, dim_3_true = self.norm.unnormalize(
            y_true[:, :, 0], y_true[:, :, 1], y_true[:, :, 2]
        )
        dim_1_pred, dim_2_pred, dim_3_pred = self.norm.unnormalize(
            y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2]
        )

        # Convert to Cartesian coordinates if necessary
        if self.cylindrical:
            dim_1_true, dim_2_true, dim_3_true = self.convert_cylindrical_to_cartesian(
                dim_1_true, dim_2_true, dim_3_true
            )
            dim_1_pred, dim_2_pred, dim_3_pred = self.convert_cylindrical_to_cartesian(
                dim_1_pred, dim_2_pred, dim_3_pred
            )

        # Compute Euclidean distance
        diff_x = dim_1_true - dim_1_pred
        diff_y = dim_2_true - dim_2_pred
        diff_z = dim_3_true - dim_3_pred

        return K.sqrt(K.square(diff_x) + K.square(diff_y) + K.square(diff_z))
    
def generate_data_redvid(df, number_of_tracks=1000, event_range=[0, -1], max_size_tracks=30, max_size_hits=450, verbose=True):
    """
    Generate data from the redvid dataset.

    Args:
        df (DataFrame): Input DataFrame containing the dataset.
        number_of_tracks (int): Number of tracks to generate. Default is 1000.
        event_range (list): Range of events to process as [start, end]. Default is [0, -1].
        max_size_tracks (int): Maximum size of tracks. Default is 30.
        max_size_hits (int): Maximum size of hits per event. Default is 450.
        verbose (bool): Whether to print progress updates. Default is True.

    Returns:
        tuple: A tuple containing tracks and hits arrays.
    """
    if event_range[1] == -1:
        event_range[1] = df["event_id"].max()

    # Initialize arrays for tracks and hits
    tracks = np.zeros((number_of_tracks, max_size_tracks + 1, 3), dtype=np.float64)
    hits = np.zeros((number_of_tracks, max_size_hits, 3), dtype=np.float64)
    tracks[:, 0, :] = np.broadcast_to([0, 0, 0.5], (tracks.shape[0], tracks.shape[2]))  # Start token

    i = 0
    breaking = False

    # Process events
    for event_id in range(*event_range):
        if breaking:
            break

        # Get hits vector, sorted by |hit_z|
        hit_vec = df.loc[df['event_id'] == event_id].reindex(
            df.loc[df['event_id'] == event_id].hit_z.abs().sort_values().index
        )[["hit_r", "hit_theta", "hit_z"]]

        # Process tracks for the current event
        for track_id in range(df[df["event_id"] == event_id]["track_id"].max() + 1):
            track_vec = df.loc[(df['event_id'] == event_id) & (df['track_id'] == track_id)][["hit_r", "hit_theta", "hit_z"]]

            tracks[i, 1:len(track_vec)+1] = track_vec
            tracks[i, len(track_vec)+1] = [0.1, 0, 0.5]  # Stop token
            hits[i, 0:len(hit_vec)] = hit_vec

            # Add first two hits of track to hits vector (if possible)
            try:
                hits[i, len(hit_vec):len(hit_vec)+3] = track_vec[:3]
            except ValueError as e:
                print("Error raised during generation, continuing:", e)
                continue

            i += 1
            if verbose and i % 1000 == 0:
                print("Generated", i, "tracks out of", number_of_tracks)

            if i == number_of_tracks:
                breaking = True
                break

    return tracks, hits


def generate_data_redvid_memeff(df, number_of_tracks=1000, event_range=[0, -1], max_size_tracks=30, max_size_hits=450, verbose=True):
    """
    Generate data from the redvid dataset in a memory-efficient way.

    Args:
        df (DataFrame): Input DataFrame containing the dataset.
        number_of_tracks (int): Number of tracks to generate. Default is 1000.
        event_range (list): Range of events to process as [start, end]. Default is [0, -1].
        max_size_tracks (int): Maximum size of tracks. Default is 30.
        max_size_hits (int): Maximum size of hits per event. Default is 450.
        verbose (bool): Whether to print progress updates. Default is True.

    Returns:
        tuple: A tuple containing tracks, hits, and track-event map arrays.
    """
    if event_range[1] == -1:
        event_range[1] = df["event_id"].max()

    # Initialize arrays for tracks and hits
    tracks = np.zeros((number_of_tracks, max_size_tracks + 1, 3), dtype=np.float64)
    track_event_map = np.zeros((number_of_tracks), dtype=np.int64)
    hits = None

    i = 0
    breaking = False

    # Process events
    for event_id in range(*event_range):
        if breaking:
            break

        # Get hits vector for the event, padded to max_size_hits
        hit_vec = df.loc[df['event_id'] == event_id].reindex(
            df.loc[df['event_id'] == event_id].hit_z.abs().sort_values().index
        )[["hit_r", "hit_theta", "hit_z"]]
        hit_vec = np.pad(hit_vec, ((0, max_size_hits - len(hit_vec)), (0, 0)), 'constant', constant_values=0)
        hit_vec = np.expand_dims(hit_vec, axis=0)

        if hits is None:
            hits = hit_vec
        else:
            hits = np.append(hits, hit_vec, axis=0)

        # Process tracks for the current event
        for track_id in range(df[df["event_id"] == event_id]["track_id"].max() + 1):
            track_vec = df.loc[(df['event_id'] == event_id) & (df['track_id'] == track_id)][["hit_r", "hit_theta", "hit_z"]]

            tracks[i, 0] = [0, 0, 0.5]  # Start token
            tracks[i, 1:len(track_vec)+1] = track_vec
            tracks[i, len(track_vec)+1] = [0.1, 0, 0.5]  # Stop token

            track_event_map[i] = event_id

            i += 1
            if verbose and i % 1000 == 0:
                print("Generated", i, "tracks out of", number_of_tracks)

            if i == number_of_tracks:
                breaking = True
                break

    return tracks, hits, track_event_map


def format_dataset(hits, tracks):
    """
    Formats the dataset into encoder and decoder inputs.

    Args:
        hits (array-like): Hit data.
        tracks (array-like): Track data.

    Returns:
        tuple: Dictionary of encoder inputs and decoder inputs, and the decoder targets.
    """
    return (
        {"encoder_inputs": hits, "decoder_inputs": tracks[:, :-1]},
        tracks[:, 1:]
    )


def make_dataset(hits, tracks, batch_size=64):
    """
    Creates a batched TensorFlow dataset.

    Args:
        hits (array-like): Hit data.
        tracks (array-like): Track data.
        batch_size (int): Batch size for the dataset. Default is 64.

    Returns:
        tf.data.Dataset: A prefetched and cached dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((hits, tracks))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(format_dataset)
    print(type(dataset))
    return dataset.prefetch(tf.data.AUTOTUNE).cache()


def make_dataset_memeff(hits, tracks, track_event_map, batch_size=64):
    """
    Creates a memory-efficient dataset using a generator for batched data.

    Args:
        hits (array-like): Hit data.
        tracks (array-like): Track data.
        track_event_map (array-like): Map of track to event IDs.
        batch_size (int): Batch size for the dataset. Default is 64.

    Returns:
        tf.data.Dataset: A prefetched and cached dataset.
    """

    def dataset_generator_memeff(hits, tracks, track_event_map):
        """
        Generator for creating a memory-efficient dataset.

        Args:
            hits (array-like): Hit data.
            tracks (array-like): Track data.
            track_event_map (array-like): Map of track to event IDs.

        Yields:
            tuple: Hits and tracks for the dataset.
        """
        i = 0
        curr_event = copy.copy(track_event_map[0])
        max_size_hits = hits.shape[1]
        
        while i < tracks.shape[0]:
            hits_to_return = hits[track_event_map[i]]

            if track_event_map[i] != curr_event:  # New event
                if (
                    np.argmax(np.all(hits_to_return == 0, axis=1)) > (max_size_hits - 5)
                    or np.argmax(np.all(hits_to_return == 0, axis=1)) == 0
                ):
                    print("WARNING: max_size_hits is too small, skipping this track")
                    i += 1
                    continue

                curr_event = track_event_map[i]
                hits_to_return[
                    np.argmax(np.all(hits_to_return == 0, axis=1)):np.argmax(np.all(hits_to_return == 0, axis=1)) + 4
                ] = tracks[i, :4]
            else:  # Same event
                hits_to_return[
                    np.argmax(np.all(hits_to_return == 0, axis=1)) - 4:np.argmax(np.all(hits_to_return == 0, axis=1))
                ] = tracks[i, :4]

            yield hits_to_return, tracks[i]
            i += 1

    dataset = tf.data.Dataset.from_generator(
        dataset_generator_memeff,
        args=[hits, tracks, track_event_map],
        output_types=(tf.float16, tf.float16),
        output_shapes=((hits.shape[1], 3), (tracks.shape[1], 3))
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(format_dataset)
    return dataset.prefetch(tf.data.AUTOTUNE).cache()


def preprocess_from_df(file_name, number_of_tracks, max_size_tracks, max_size_hits, batch_size=64):
    """
    Preprocess data from a DataFrame and create train and validation datasets.

    Args:
        file_name (str): Path to the CSV file.
        number_of_tracks (int): Number of tracks to generate.
        max_size_tracks (int): Maximum size of tracks.
        max_size_hits (int): Maximum size of hits.
        batch_size (int): Batch size for datasets. Default is 64.

    Returns:
        tuple: Training dataset, validation dataset, and the normalizer object.
    """
    df = pd.read_csv(file_name, delimiter=';')

    # Normalize data
    norm = Normalizer(
        df["hit_r"].min(), df["hit_r"].max(),
        df["hit_theta"].min(), df["hit_theta"].max(),
        df["hit_z"].min(), df["hit_z"].max()
    )
    df["hit_r"], df["hit_theta"], df["hit_z"] = norm.normalize(
        df["hit_r"], df["hit_theta"], df["hit_z"]
    )

    # Generate data
    tracks, hits = generate_data(
        df,
        max_size_hits=max_size_hits,
        number_of_tracks=number_of_tracks,
        max_size_tracks=max_size_tracks
    )

    # Train-validation split
    num_val_samples = int(0.20 * number_of_tracks)
    val_tracks = tracks[:num_val_samples]
    val_hits = hits[:num_val_samples]
    train_tracks = tracks[num_val_samples:]
    train_hits = hits[num_val_samples:]

    train_ds = make_dataset(train_hits, train_tracks, batch_size=batch_size)
    val_ds = make_dataset(val_hits, val_tracks, batch_size=batch_size)

    return train_ds, val_ds, norm

def preprocess_from_parsed_data(parsed_data_dir, max_size_tracks, max_size_hits, batch_size=64):
    """
    Preprocess data

    arguments:
        parsed_data_dir (str): parsed (not normalized!) numpy arrays tracks and hits data dir
        max_size_tracks (int): maximum size of tracks
        max_size_hits (int): maximum size of hits
        batch_size (int): batch size
    """

    tracks = np.load(parsed_data_dir + "tracks.npy")
    hits = np.load(parsed_data_dir + "hits.npy")

    dim_1_min = np.min(hits[:, :, 0])
    dim_1_max = np.max(hits[:, :, 0])

    dim_2_min = np.min(hits[:, :, 1])
    dim_2_max = np.max(hits[:, :, 1])

    dim_3_min = np.min(hits[:, :, 2])
    dim_3_max = np.max(hits[:, :, 2])

    if dim_1_max > 1 or dim_2_max > 1 or dim_3_max > 1: # Not normalized
        print("WARNING: It looks like your data is not normalized, please normalize your parsed data beforehand!")

    if dim_1_min < 0 or dim_2_min < 0 or dim_3_min < 0: # Not normalized
        print("WARNING: It looks like your data is not normalized, please normalize your parsed data beforehand!")

    f = open(parsed_data_dir + 'extrema.json')
    extrema = json.load(f)

    norm = Normalizer(float(extrema["dim_1_min"]), float(extrema["dim_1_max"]), float(extrema["dim_2_min"]), float(extrema["dim_2_max"]), float(extrema["dim_3_min"]), float(extrema["dim_3_max"]))

    number_of_tracks = tracks.shape[0]

    #train val split
    num_val_samples = int(0.20 * number_of_tracks)

    val_tracks = tracks[:num_val_samples]
    val_hits = hits[:num_val_samples]

    train_tracks = tracks[num_val_samples:]
    train_hits = hits[num_val_samples:]

    train_ds = make_dataset(train_hits, train_tracks, batch_size=batch_size)
    val_ds = make_dataset(val_hits, val_tracks, batch_size=batch_size)

    return train_ds, val_ds, norm

def preprocess_from_parsed_data_memeff(parsed_data_dir, max_size_tracks, max_size_hits, batch_size=64):
    """
    Preprocess data

    arguments:
        parsed_data_dir (str): parsed (not normalized!) numpy arrays tracks and hits data dir
        max_size_tracks (int): maximum size of tracks
        max_size_hits (int): maximum size of hits
        batch_size (int): batch size
    """

    tracks = np.load(parsed_data_dir + "tracks.npy")
    hits = np.load(parsed_data_dir + "hits.npy")
    track_event_map = np.load(parsed_data_dir + "track_event_map.npy").astype(np.int32)

    dim_1_min = np.min(hits[:, :, 0])
    dim_1_max = np.max(hits[:, :, 0])

    dim_2_min = np.min(hits[:, :, 1])
    dim_2_max = np.max(hits[:, :, 1])

    dim_3_min = np.min(hits[:, :, 2])
    dim_3_max = np.max(hits[:, :, 2])

    if dim_1_max > 1 or dim_2_max > 1 or dim_3_max > 1: # Not normalized
        print("WARNING: It looks like your data is not normalized, please normalize your parsed data beforehand!")

    if dim_1_min < 0 or dim_2_min < 0 or dim_3_min < 0: # Not normalized
        print("WARNING: It looks like your data is not normalized, please normalize your parsed data beforehand!")

    f = open(parsed_data_dir + 'extrema.json')
    extrema = json.load(f)

    norm = Normalizer(float(extrema["dim_1_min"]), float(extrema["dim_1_max"]), float(extrema["dim_2_min"]), float(extrema["dim_2_max"]), float(extrema["dim_3_min"]), float(extrema["dim_3_max"]))

    number_of_events = hits.shape[0]

    #train val split
    num_val_events = int(0.20 * number_of_events)

    print(track_event_map.shape)
    print(hits.shape)
    print(tracks.shape)

    val_track_event_map = track_event_map[track_event_map < hits[:num_val_events].shape[0]]
    val_tracks = tracks[track_event_map < hits[:num_val_events].shape[0]]
    val_hits = hits[:num_val_events]

    train_track_event_map = track_event_map[track_event_map >= hits[:num_val_events].shape[0]] - num_val_events
    train_tracks = tracks[track_event_map >= hits[:num_val_events].shape[0]]
    train_hits = hits[num_val_events:]

    train_ds = make_dataset_memeff(train_hits, train_tracks, train_track_event_map, batch_size=batch_size)
    val_ds = make_dataset_memeff(val_hits, val_tracks, val_track_event_map, batch_size=batch_size)

    return train_ds, val_ds, norm

def decode_track(input_hits, input_track, transformer, norm, max_size_tracks=30, cylindrical=True):
    """
    Decodes a track using a transformer model and finds the closest hits in the input data.

    Args:
        input_hits (np.array): Input hits array of shape (1, max_size_hits, 3).
        input_track (np.array): Input track array of shape (1, max_size_tracks, 3).
        transformer (tf.keras.Model): Transformer model used for prediction.
        norm (Normalizer): Normalizer instance for unnormalizing data.
        max_size_tracks (int): Maximum number of hits per track. Default is 30.
        cylindrical (bool): Whether the coordinates are in cylindrical form. Default is True.

    Returns:
        np.array: Decoded output track of shape (1, max_size_tracks, 3).
    """
    # Initialize output track and copy the start tokens from the input
    output_track = np.zeros((1, max_size_tracks, 3), dtype="float32")
    output_track[:, :4, :] = input_track[:, :4, :]
    output_track = tf.convert_to_tensor(output_track)

    for i in range(3, max_size_tracks - 2):
        # Predict the next timestep using the transformer model
        predictions = transformer([input_hits, output_track])
        predictions_timestep = predictions[0, i, :]

        # Unnormalize the predicted timestep and input hits
        unnormalized_predictions = np.zeros(predictions_timestep.shape)
        unnormalized_predictions[0], unnormalized_predictions[1], unnormalized_predictions[2] = norm.unnormalize(
            predictions_timestep[0], predictions_timestep[1], predictions_timestep[2]
        )

        input_hits_no_zeros = input_hits[0, :, :][~np.all(input_hits[0, :, :] == 0, axis=1)]
        input_hits_no_zeros = np.append(input_hits_no_zeros, [[0.1, 0, 0.5]], axis=0)

        unnormalized_hits = np.zeros(input_hits_no_zeros.shape)
        unnormalized_hits[:, 0], unnormalized_hits[:, 1], unnormalized_hits[:, 2] = norm.unnormalize(
            input_hits_no_zeros[:, 0], input_hits_no_zeros[:, 1], input_hits_no_zeros[:, 2]
        )

        # Convert to Cartesian coordinates if required
        if cylindrical:
            unnormalized_predictions_cart = np.zeros(predictions_timestep.shape)
            unnormalized_hits_cart = np.zeros(input_hits_no_zeros.shape)

            unnormalized_predictions_cart[0], unnormalized_predictions_cart[1], unnormalized_predictions_cart[2] = \
                convert_cylindrical_to_cartesian(
                    unnormalized_predictions[0], unnormalized_predictions[1], unnormalized_predictions[2]
                )

            unnormalized_hits_cart[:, 0], unnormalized_hits_cart[:, 1], unnormalized_hits_cart[:, 2] = \
                convert_cylindrical_to_cartesian(
                    unnormalized_hits[:, 0], unnormalized_hits[:, 1], unnormalized_hits[:, 2]
                )
        else:
            unnormalized_predictions_cart = unnormalized_predictions
            unnormalized_hits_cart = unnormalized_hits

        # Get the true hit at the next timestep
        true_hit = input_track[:, i + 1, :]
        if np.isclose(true_hit[0][2], 0, atol=1e-1) or \
           np.isclose(true_hit, np.array([[0.1, 0, 0.5]]), atol=1e-3).all(axis=1):
            break  # Stop if the true hit is close to the end token

        # Calculate Euclidean distances and find the closest hit
        diff = unnormalized_hits_cart - unnormalized_predictions_cart
        euclid_dist = np.sqrt(np.sum(diff * diff, axis=1))
        best_hit = input_hits_no_zeros[euclid_dist.argmin()]

        # Update the output track with the best hit
        other = np.zeros((1, max_size_tracks, 3))
        other[0, i + 1] += best_hit

        mask = np.ones((1, max_size_tracks, 3))
        mask[0, i + 1, :] -= 1
        output_track = output_track * mask + other * (1 - mask)

        # Stop if the closest hit is the end token
        if euclid_dist.argmin() == (input_hits_no_zeros.shape[0] - 1):
            break

    return output_track

def predict_track(input_hits, input_track, transformer, norm, max_size_tracks=30, cylindrical=True):
    """
    Predicts a track using a transformer model and evaluates performance based on hit selection.

    Args:
        input_hits (np.array): Input hits array of shape (1, max_size_hits, 3).
        input_track (np.array): Input track array of shape (1, max_size_tracks, 3).
        transformer (tf.keras.Model): Transformer model used for prediction.
        norm (Normalizer): Normalizer instance for unnormalizing data.
        max_size_tracks (int): Maximum number of hits per track. Default is 30.
        cylindrical (bool): Whether the coordinates are in cylindrical form. Default is True.

    Returns:
        tuple: Counts of correctly chosen hits, hits within tolerance, and the track length.
    """
    correctly_chosen_hits = 0
    hits_within_tolerance = 0
    track_length = 0

    # Initialize output track and copy start tokens from input
    output_track = np.zeros((1, max_size_tracks, 3), dtype="float16")
    output_track[:, :4, :] = input_track[:, :4, :]
    output_track = tf.convert_to_tensor(output_track)

    for i in range(3, max_size_tracks - 2):
        # Predict the next timestep using the transformer model
        predictions = transformer([input_hits, output_track])
        predictions_timestep = predictions[0, i, :]

        # Unnormalize predictions and hits, removing trailing zeros
        unnormalized_predictions = np.zeros(predictions_timestep.shape)
        unnormalized_predictions[0], unnormalized_predictions[1], unnormalized_predictions[2] = norm.unnormalize(
            predictions_timestep[0], predictions_timestep[1], predictions_timestep[2]
        )

        input_hits_no_zeros = input_hits[0, :, :][~np.all(input_hits[0, :, :] == 0, axis=1)]
        input_hits_no_zeros = np.append(input_hits_no_zeros, [[0.1, 0, 0.5]], axis=0)

        unnormalized_hits = np.zeros(input_hits_no_zeros.shape)
        unnormalized_hits[:, 0], unnormalized_hits[:, 1], unnormalized_hits[:, 2] = norm.unnormalize(
            input_hits_no_zeros[:, 0], input_hits_no_zeros[:, 1], input_hits_no_zeros[:, 2]
        )

        # Convert to Cartesian coordinates if needed
        if cylindrical:
            unnormalized_predictions_cart = np.zeros(predictions_timestep.shape)
            unnormalized_hits_cart = np.zeros(input_hits_no_zeros.shape)

            unnormalized_predictions_cart[0], unnormalized_predictions_cart[1], unnormalized_predictions_cart[2] = \
                convert_cylindrical_to_cartesian(
                    unnormalized_predictions[0], unnormalized_predictions[1], unnormalized_predictions[2]
                )

            unnormalized_hits_cart[:, 0], unnormalized_hits_cart[:, 1], unnormalized_hits_cart[:, 2] = \
                convert_cylindrical_to_cartesian(
                    unnormalized_hits[:, 0], unnormalized_hits[:, 1], unnormalized_hits[:, 2]
                )
        else:
            unnormalized_predictions_cart = unnormalized_predictions
            unnormalized_hits_cart = unnormalized_hits

        # Calculate Euclidean distance and find the closest hit
        diff = unnormalized_hits_cart - unnormalized_predictions_cart
        euclid_dist = np.sqrt(np.sum(diff * diff, axis=1))
        chosen_hit = input_hits_no_zeros[euclid_dist.argmin()]
        best_hit = input_track[:, i + 1, :]

        # Stop if the best hit is close to the end token
        if np.isclose(best_hit[0][2], 0, atol=1e-1) or \
           np.isclose(best_hit, np.array([[0.1, 0, 0.5]]), atol=1e-3).all(axis=1):
            break

        # Increment counters for hit evaluation
        correctly_chosen_hits += np.isclose(chosen_hit, best_hit, atol=1e-3).all(axis=1)
        hits_within_tolerance += np.isclose(predictions_timestep, best_hit, atol=5e-2).all(axis=1)
        track_length += 1

        # Update output track with the best hit
        other = np.zeros((1, max_size_tracks, 3))
        other[0, i + 1] += best_hit

        mask = np.ones((1, max_size_tracks, 3))
        mask[0, i + 1, :] -= 1
        output_track = output_track * mask + other * (1 - mask)

    return correctly_chosen_hits, hits_within_tolerance, track_length