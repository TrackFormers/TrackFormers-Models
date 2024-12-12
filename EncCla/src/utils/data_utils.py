import torch
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import logging


def load_dataloader(config, device, mode="all"):
    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["dataloader_num_workers"]

    loaders = {}

    if mode == "train" or mode == "all":
        shuffle = config["training"]["shuffle"]
        logging.info("Loading train data with DataLoader")
        train_dataset = ForwardPassDataset(data_dir, config["data"]["train_file"])
        val_dataset = ForwardPassDataset(data_dir, config["data"]["val_file"])
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    if mode == "eval" or mode == "all":
        test_dataset = ForwardPassDataset(data_dir, config["data"]["test_file"])
        test_helper_dataset = ScoringHelperDataset(
            data_dir, config["data"]["test_helperfile"]
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["test_helper"] = DataLoader(
            test_helper_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return loaders


def load_truths(config):
    data_dir = config["data"]["data_dir"]
    test_truth_filename = config["data"]["test_truthfile"]
    test_truth_filepath = os.path.join(data_dir, test_truth_filename)
    return pd.read_csv(test_truth_filepath)


# dynamically padding data based on the max seq_length of the batch
def collate_fn(batch):
    dat1, dat2 = zip(*batch)
    dat1_padded = pad_sequence(dat1, batch_first=True, padding_value=0.0)
    dat2_padded = pad_sequence(dat2, batch_first=True, padding_value=0)
    return dat1_padded, dat2_padded


class ForwardPassDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.coords, self.labels = torch.load(file_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        coords_tensor = torch.tensor(self.coords[idx], dtype=torch.float)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return coords_tensor, labels_tensor


class ScoringHelperDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.hit_ids, self.event_ids = torch.load(file_path)

    def __len__(self):
        return len(self.hit_ids)

    def __getitem__(self, idx):
        hit_ids_tensor = torch.tensor(self.hit_ids[idx], dtype=torch.long)
        event_ids_tensor = torch.tensor(self.event_ids[idx], dtype=torch.long)
        return hit_ids_tensor, event_ids_tensor
