import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from data_processing.dataset import PAD_TOKEN


class ClustersDataset(Dataset):
    '''
    Dataset class for the clusters of hits. Analogous to the HitsDataset, but 
    since the hits are already grouped (based on event ID and particle ID), we
    do not make use of labels.
    '''
    def __init__(self, hit_clusters, track_params):
        self.hit_clusters = hit_clusters
        self.track_params = track_params
        self.total_clusters = self.__len__()

    def __len__(self):
        return self.hit_clusters.shape[0]

    def __getitem__(self, idx):
        return idx, self.hit_clusters[idx], self.track_params[idx]


def load_data_for_refiner(data_path, normalize=False):
    '''
    Identical to the load_trackml_data function from trackml_data.py. However, here
    the data is grouped by the combination of event ID and particle ID, and not 
    only by event ID.
    '''
    data = pd.read_csv(data_path)

    if normalize:
        for col in ["x", "y", "z"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event_particle = shuffled_data.groupby(['event_id', 'particle_id'])
    max_num_hits = 50

    def extract_input(rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        hit_data = rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(hit_data, [(0, max_num_hits-len(rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_output(rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = rows[["px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,0]**2 + event_track_params_data[:,1]**2 + event_track_params_data[:,2]**2)
        theta = np.arccos(event_track_params_data[:,2]/p)
        phi = np.arctan2(event_track_params_data[:,1], event_track_params_data[:,0])
        processed_event_track_params_data = np.column_stack([theta, np.sin(phi), np.cos(phi), event_track_params_data[:,3]])
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    grouped_hits_data = data_grouped_by_event_particle.apply(extract_input)
    grouped_track_params_data = data_grouped_by_event_particle.apply(extract_output)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))

    return hits_data, track_params_data