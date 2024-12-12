import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd

PAD_TOKEN = -1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HitsDataset(Dataset):
    '''
    Dataset class for the detector data, i.e. the hit coordinates, their track
    parameters, the particles they belong to.
    '''

    def __init__(self, hits_data, track_params_data=None, class_data=None):
        self.hits_data = hits_data.to(DEVICE)
        self.track_params_data = track_params_data.to(DEVICE)
        self.class_data = class_data.to(DEVICE)
        self.total_events = self.__len__()

    def __len__(self):
        return self.hits_data.shape[0]

    def __getitem__(self, idx):
        return idx, self.hits_data[idx], self.track_params_data[idx], self.class_data[idx]

def get_dataloaders(dataset, train_frac, valid_frac, test_frac, batch_size):
    train_set, valid_set, test_set = random_split(dataset, [train_frac, valid_frac, test_frac], generator=torch.Generator().manual_seed(37))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

def load_linear_2d_data(data_path, max_num_hits=None):
    '''
    Function for reading .csv file with 2D linear coordinates and creating tensors
    containing the hits and ground truth data from it.
    Note: this function does not pad the data and assumes all events have the 
    same number of hits.
    '''
    full_data = pd.read_csv(data_path, sep=",")
    full_data = full_data.sample(frac=1)
    
    data_grouped_by_event = full_data.groupby("event_id")

    grouped_hits_data = data_grouped_by_event.apply(lambda event_rows: event_rows[["hit_x", "hit_y"]].to_numpy(dtype=np.float32))
    grouped_track_params_data = data_grouped_by_event.apply(lambda event_rows: event_rows["slope"].to_numpy(dtype=np.float32))
    grouped_hit_classes_data = data_grouped_by_event.apply(lambda event_rows: event_rows["track_id"].to_numpy(dtype=int))

    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values)).unsqueeze(2)
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data

def load_linear_3d_data(data_path, max_num_hits):
    '''
    Function for reading .csv file with 3D linear coordinates and creating tensors
    containing the hits and ground truth data from it.
    max_num_hits denotes the size of the largest event, to pad the other events
    up to.
    '''
    full_data = pd.read_csv(data_path, sep=";")
    full_data = full_data.sample(frac=1)
    
    data_grouped_by_event = full_data.groupby("event_id")

    def extract_hits_data(event_rows):
        event_hit_data = event_rows[["hit_r", "hit_theta", "hit_z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)
    
    def extract_track_params_data(event_rows):
        event_track_params_data = event_rows[["theta_d", "z_d", "r_d"]].to_numpy(dtype=np.float32)
        event_angles_theta = np.pi/2. - np.arctan2(event_track_params_data[:, 1], event_track_params_data[:, 2])
        processed_event_track_params_data = np.column_stack((event_angles_theta, np.sin(event_track_params_data[:,0]), np.cos(event_track_params_data[:,0])))
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)
    
    def extract_hit_classes_data(event_rows):
        event_hit_classes_data = event_rows["track_id"].to_numpy(dtype=int)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows))], "constant", constant_values=PAD_TOKEN)

    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data

def load_curved_3d_data(data_path, max_num_hits):
    '''
    Function for reading .csv file with 3D helical coordinates and creating tensors
    containing the hits and ground truth data from it.
    max_num_hits denotes the size of the largest event, to pad the other events
    up to.
    '''
    full_data = pd.read_csv(data_path, sep=";")
    full_data = full_data.sample(frac=1)
    
    data_grouped_by_event = full_data.groupby("event_id")

    def extract_hits_data(event_rows):
        event_hit_data = event_rows[["hit_r", "hit_theta", "hit_z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)
    
    def extract_track_params_data(event_rows):
        event_track_params_data = event_rows[["radial_coeff","pitch_coeff","azimuthal_coeff"]].to_numpy(dtype=np.float32)
        return np.pad(event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)
    
    def extract_hit_classes_data(event_rows):
        event_hit_classes_data = event_rows["track_id"].to_numpy(dtype=int)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows))], "constant", constant_values=PAD_TOKEN)

    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data
