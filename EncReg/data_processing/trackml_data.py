import torch
import numpy as np
import pandas as pd
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from data_processing.dataset import PAD_TOKEN


def take_inner_detector_trackml_data(event_id, file_name):
    '''
    Parse the three TrackML files per event and write the data to file file_name,
    such that only hits from the inner detector are kept.
    '''
    try:
        hits_data = pd.read_csv(f'event0000{event_id}-hits.csv')
        particles_data = pd.read_csv(f'event0000{event_id}-particles.csv')
        truth_data = pd.read_csv(f'event0000{event_id}-truth.csv')
    except:
        print('File does not exist')
        return

    # Take only hits from inner detector
    indices = hits_data['volume_id'] <= 9
    hits_data = hits_data[indices]
    truth_data = truth_data[indices]

    indices = particles_data['particle_id'].isin(truth_data['particle_id'])
    particles_data = particles_data[indices]

    # Merge hit, truth and particle dataframes into a single one with the relevant variables
    hits_data = hits_data[["hit_id", "x", "y", "z", "volume_id"]]
    hits_data.insert(0, column="particle_id", value=truth_data["particle_id"].values)
    hits_data.insert(0, column="weight", value=truth_data["weight"].values)

    merged_data = hits_data.merge(truth_data, left_on='hit_id', right_on='hit_id')
    merged_data = merged_data.merge(particles_data, left_on='particle_id_x', right_on='particle_id')

    final_data = merged_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x"]]
    final_data['event_id'] = event_id

    # Write the event to a file
    final_data.to_csv(file_name, mode='a', index=False, header=False)

def transform_trackml_data(event_id, additional_id, min_nr_particles, max_nr_particles):
    '''
    Parse the three TrackML files per event and write the data to file,
    such that only p particles with their hits are sampled, where 
    p in [min_nr_particles, max_nr_particles]. Sample 5 times, effectively
    creating 5 smaller events out of the original one.
    '''
    try:
        hits_data = pd.read_csv(f'event0000{event_id}-hits.csv')
        particles_data = pd.read_csv(f'event0000{event_id}-particles.csv')
        truth_data = pd.read_csv(f'event0000{event_id}-truth.csv')
    except:
        print('File does not exist')
        return

    for i in range(5):
        # Take a subset of the particles
        nr_particles_in_event = random.randint(min_nr_particles, max_nr_particles)
        sampled_particle_ids = particles_data['particle_id'].unique().tolist()[:nr_particles_in_event]
        indices = particles_data['particle_id'].isin(sampled_particle_ids)
        particles_data = particles_data[indices]

        indices = truth_data['particle_id'].isin(particles_data["particle_id"])
        truth_data = truth_data[indices]
        hits_data = hits_data[indices]

        # Merge hit, truth and particle dataframes into a single one with the relevant variables
        hits_data = hits_data[["hit_id", "x", "y", "z", "volume_id"]]
        hits_data.insert(0, column="particle_id", value=truth_data["particle_id"].values)
        hits_data.insert(0, column="weight", value=truth_data["weight"].values)

        merged_data = hits_data.merge(truth_data, left_on='hit_id', right_on='hit_id')
        merged_data = merged_data.merge(particles_data, left_on='particle_id_x', right_on='particle_id')

        final_data = merged_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x"]]
        final_data['event_id'] = additional_id + i # Ensures each subevent has different event ID

        # Write the event to a file
        final_data.to_csv(f'trackml_{min_nr_particles}to{max_nr_particles}tracks.csv', mode='a', index=False, header=False)


def load_trackml_data(data, max_num_hits, normalize=True, chunking=False):
    '''
    Function for reading .csv file with TrackML data and creating tensors
    containing the hits and ground truth information from it.
    max_num_hits denotes the size of the largest event, to pad the other events
    up to. normalize decides whether the data will be normalized first. 
    chunking allows for reading .csv files in chunks.
    '''
    if not chunking:
        data = pd.read_csv(data)

    # Normalize the data if applicable
    if normalize:
        for col in ["x", "y", "z", "px", "py", "pz", "q"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event = shuffled_data.groupby("event_id")

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,0]**2 + event_track_params_data[:,1]**2 + event_track_params_data[:,2]**2)
        theta = np.arccos(event_track_params_data[:,2]/p)
        phi = np.arctan2(event_track_params_data[:,1], event_track_params_data[:,0])
        processed_event_track_params_data = np.column_stack([theta, np.sin(phi), np.cos(phi), event_track_params_data[:,3]])
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data


def load_trackml_data_pt(data, max_num_hits, pt_min, pt_max, normalize=False, chunking=False):
    '''
    Function for reading .csv file with TrackML data and creating tensors
    containing the hits and ground truth information from it.
    max_num_hits denotes the size of the largest event, to pad the other events
    up to. normalize decides whether the data will be normalized first. 
    chunking allows for reading .csv files in chunks.
    pt_min and pt_max determine the range of the momentum p that the hits must
    fall into to be part of the dataset.
    '''
    if not chunking:
        data = pd.read_csv(data)

    # Normalize the data if applicable
    if normalize:
        for col in ["x", "y", "z", "px", "py", "pz", "q"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    data['p'] = np.sqrt(data["px"]**2 + data["py"]**2 + data["pz"]**2)
    data = data[np.logical_and(data['p'] >= pt_min, data['p'] < pt_max)]

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event = shuffled_data.groupby("event_id")

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,0]**2 + event_track_params_data[:,1]**2 + event_track_params_data[:,2]**2)
        theta = np.arccos(event_track_params_data[:,2]/p)
        phi = np.arctan2(event_track_params_data[:,1], event_track_params_data[:,0])
        processed_event_track_params_data = np.column_stack([theta, np.sin(phi), np.cos(phi), event_track_params_data[:,3]])
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data


if __name__ == "__main__":
    # For creating subset data
    id = 0
    for i in range(21000, 30000):
        transform_trackml_data(i, id, 10, 50)
        id += 5

    # For inner detector data:
    # for i in range(21000, 30000):
    #     take_inner_detector_trackml_data(i, 'trackml_inner_detector.csv')
