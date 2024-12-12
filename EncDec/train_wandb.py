import wandb
import argparse
import os
import functools
import traceback
from datetime import datetime
import sys

from train import train

def train_wandb(config=None, 
                number_of_tracks=1000, 
                max_size_tracks=30, 
                max_size_hits=450, 
                infile_name='data.csv', 
                slug='training_temp', 
                parsed_data_dir=None,
                output_foldername="./output",
                memeff=False):
    """
    Train the model with Weights and Biases integration.

    Args:
        config (dict, optional): Dictionary with hyperparameters, set by
                                 Weights and Biases. Defaults to None.
        number_of_tracks (int): Number of tracks to train on.
        max_size_tracks (int): Maximum size of tracks.
        max_size_hits (int): Maximum size of hits.
        infile_name (str): Input file name.
        slug (str): Identifier for the training run.
        parsed_data_dir (str): Directory for parsed data.
        output_foldername (str): Base directory for outputs.
        memeff (bool): Enable memory efficiency mode.
    """
    with wandb.init(settings=wandb.Settings(start_method="fork")):
        config = wandb.config

        # Create directories for saving models and logs
        time_string = "-" + datetime.now().strftime("%d-%mT%H-%M-%S")
        model_save_path = os.path.join(
            output_foldername, slug, wandb.run.name + time_string, "saved_models"
        )
        log_save_path = os.path.join(
            output_foldername, slug, wandb.run.name + time_string, "logs"
        )
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_save_path, exist_ok=True)
        
        save_dir = os.path.join(model_save_path, "save_model_h5")
        log_dir = log_save_path

        try:
            # Start training with the provided configuration
            train(
                number_of_tracks=number_of_tracks,
                max_size_tracks=max_size_tracks,
                max_size_hits=max_size_hits,
                hyperparameter_config=config,
                infile_name=infile_name,
                parsed_data_dir=parsed_data_dir,
                save_dir=save_dir,
                log_dir=log_dir,
                wandb=True,
                memeff=memeff
            )
        except Exception:
            # Print traceback and re-raise exception on failure
            print(traceback.print_exc())
            print(sys.exc_info()[2])
            raise Exception

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with Wandb')
    parser.add_argument('--hyperparameter_config_path', 
                        type=str, 
                        default="./configs/hyperparameters/3d_noisy_curved_10to50.json", 
                        help='path to the hyperparameter config file')
    parser.add_argument(
        "--output_foldername",
        type=str,
        nargs="?",
        help="Base directory for outputs",
        default="./output",
    )
    parser.add_argument('--number_of_tracks', type=int, default=None, help='number of tracks')
    parser.add_argument('--max_size_tracks', type=int, default=30, help='maximum size of tracks')
    parser.add_argument('--max_size_hits', type=int, default=200, help='maximum size of hits')
    parser.add_argument('--infile_name', 
                        type=str, 
                        default=None, 
                        help='input file name')
    parser.add_argument('--slug', type=str, default='training_temp', help='identifier for training run')
    parser.add_argument('--parsed_data_dir', 
                        type=str, 
                        default=None, help='parsed numpy arrays tracks and hits data directory')
    parser.add_argument('--memeff', action='store_true', help='enable memory efficiency mode')
    args = parser.parse_args()

    if args.hyperparameter_config_path is not None:
        import json
        with open(args.hyperparameter_config_path) as f:
            hyperparameter_config = json.load(f)

    del args.hyperparameter_config_path

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    sweep_id = wandb.sweep(
        hyperparameter_config,
        project=args.slug,
        entity="zeff020"
    )

    args = vars(args)

    # Start Wandb agent with the sweep configuration
    wandb.agent(sweep_id, functools.partial(train_wandb, **args), count=50)
    wandb.finish()