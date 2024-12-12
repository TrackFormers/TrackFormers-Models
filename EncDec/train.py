import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import argparse
from utils import preprocess_from_df, preprocess_from_parsed_data, preprocess_from_parsed_data_memeff, Euclid_distance_metric
from models import create_model
from datetime import datetime
import os
from wandb.keras import WandbCallback, WandbMetricsLogger

__all__ = ["train"]

def train(number_of_tracks=None, 
          max_size_tracks=30, 
          max_size_hits=450, 
          infile_name=None, 
          save_dir='saved_models/training_temp', 
          parsed_data_dir=None, 
          hyperparameter_config=None,
          log_dir='logs/training_temp',
          wandb=False,
          memeff=False):
    """
    Construct and subsequently train the model.

    arguments:
        number_of_tracks (int): number of tracks
        max_size_tracks (int): maximum size of tracks
        max_size_hits (int): maximum size of hits
        infile_name (str): file name
        save_dir (str): save directory
        parsed_data_dir (str): parsed numpy arrays tracks and hits data dir
        hyperparameter_config (dict): dictionary with hyperparameters
        log_dir (str): log directory
        wandb (bool): whether to use wandb
        memeff (bool): whether to use memory efficiency mode
    """
    # Print training configuration
    if number_of_tracks:
        print("Number of tracks: ", number_of_tracks)
    print("Maximum size of tracks: ", max_size_tracks)
    print("Maximum size of hits: ", max_size_hits)
    if infile_name:
        print("Input filename: ", infile_name)
    print("Save directory: ", save_dir)
    if parsed_data_dir:
        print("Parsed data directory: ", parsed_data_dir)
    print("Hyperparameter config: ", hyperparameter_config)
    print("Log directory: ", log_dir)
    print("Wandb: ", wandb)

    hyperparameter_config = dict(hyperparameter_config)

    print("INFO: loading data")

    if parsed_data_dir is None and infile_name is None:
        raise ValueError("either parsed_data_dir or infile_name must be specified")

    # Load and preprocess data
    if parsed_data_dir is None:
        if number_of_tracks is None:
            raise ValueError("number_of_tracks must be specified if parsed_data_dir is not specified")
        train_ds, val_ds, norm = preprocess_from_df(
            infile_name, number_of_tracks, max_size_tracks, max_size_hits, 
            batch_size=hyperparameter_config["batch_size"]
        )
    else:
        if memeff:
            train_ds, val_ds, norm = preprocess_from_parsed_data_memeff(
                parsed_data_dir, max_size_tracks, max_size_hits, 
                batch_size=hyperparameter_config["batch_size"]
            )
        else:
            train_ds, val_ds, norm = preprocess_from_parsed_data(
                parsed_data_dir, max_size_tracks, max_size_hits, 
                batch_size=hyperparameter_config["batch_size"]
            )
    
    # Display shapes of inputs and targets
    for inputs, targets in train_ds.take(1):
        print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
        print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
        print(f"targets.shape: {targets.shape}")
    
    print("INFO: done loading data")

    # Build the Transformer model
    transformer = create_model(
        hyperparameter_config["embed_dim"], 
        hyperparameter_config["dense_dim"], 
        hyperparameter_config["num_heads"], 
        num_induce=hyperparameter_config["num_induce"], 
        max_size_hits=max_size_hits, 
        max_size_tracks=max_size_tracks, 
        batch_size=hyperparameter_config["batch_size"], 
        num_stacks=hyperparameter_config["num_stacks"]
    )

    checkpoint_path = os.path.join(save_dir, "cp.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Configure learning rate schedule
    if hyperparameter_config["learning_rate_schedule"] == "constant":
        lr = hyperparameter_config["learning_rate"]
    elif hyperparameter_config["learning_rate_schedule"] == "cosine":
        lr = tf.keras.optimizers.schedules.CosineDecay(
            hyperparameter_config["learning_rate"], 
            hyperparameter_config["decay_steps"], 
            alpha=hyperparameter_config["alpha"]
        )
    elif hyperparameter_config["learning_rate_schedule"] == "cosinerestarts":
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            hyperparameter_config["learning_rate"], 
            hyperparameter_config["decay_steps"], 
            alpha=hyperparameter_config["alpha"]
        )
    elif hyperparameter_config["learning_rate_schedule"] == "exponential":
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            hyperparameter_config["learning_rate"], 
            hyperparameter_config["decay_steps"], 
            hyperparameter_config["alpha"]
        )

    # Select optimizer
    if hyperparameter_config["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif hyperparameter_config["optimizer"] == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    # Select distance metric based on dataset
    if hyperparameter_config["dataset"] == "trackml":
        print("Using Euclidean distance metric without cylindrical coordinates")
        dist = Euclid_distance_metric(norm, cylindrical=False)
    else:
        dist = Euclid_distance_metric(norm, cylindrical=True)

    transformer.summary()
    transformer.compile(
        optimizer, loss="MSE", metrics=[dist]
    )

    # Set up callbacks for training
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, mode="min", restore_best_weights=True
    )

    callbacks = [cp_callback, tensorboard_callback, early_stopping_callback]

    if wandb:
        callbacks.append([WandbCallback(save_model=False), WandbMetricsLogger()])

    # Start training the model
    transformer.fit(
        train_ds, 
        epochs=hyperparameter_config["epochs"], 
        validation_data=val_ds, 
        callbacks=callbacks, 
        use_multiprocessing=True, 
        workers=8
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--number_of_tracks', type=int, default=1000, help='number of tracks')
    parser.add_argument('--max_size_tracks', type=int, default=30, help='maximum size of tracks')
    parser.add_argument('--max_size_hits', type=int, default=450, help='maximum size of hits')
    parser.add_argument('--infile_name', type=str, default='data.csv', help='file name')
    parser.add_argument('--save_dir', type=str, default='saved_models/training_temp', help='save directory')
    parser.add_argument('--parsed_data_dir', type=str, default=None, help='parsed numpy arrays tracks and hits data dir')
    parser.add_argument('--hyperparameter_config', type=str, default=None, help='json string with hyperparameter config')
    parser.add_argument('--log_dir', type=str, default='logs/training_temp', help='log directory')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--memeff', action='store_true', help='whether to use memory efficiency mode')
    args = parser.parse_args()

    if args.hyperparameter_config is not None:
        import json
        args.hyperparameter_config = json.loads(args.hyperparameter_config)

    train(**args)