import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np



class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        # Load samples
        x_data = dataset["samples"]

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None

        self.len = x_data.shape[0]

    def get_labels(self):
        return self.y_data

    def __getitem__(self, index):
        sample = {
            'samples': self.x_data[index].squeeze(-1),
            'labels': int(self.y_data[index])
        }

        return sample

    def __len__(self):
        return self.len


def data_generator(data_path, data_type, hparams):
    # original
    train_dataset = torch.load(os.path.join(data_path, data_type, f"train.pt"))
    val_dataset = torch.load(os.path.join(data_path, data_type, f"val.pt"))
    test_dataset = torch.load(os.path.join(data_path, data_type, f"test.pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    cw = train_dataset.y_data.numpy().tolist()
    cw_dict = {}
    for i in range(len(np.unique(train_dataset.y_data.numpy()))):
        cw_dict[i] = cw.count(i)
    # print(cw_dict)

    # Dataloaders
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader, test_loader, get_class_weight(cw_dict)


import math


def get_class_weight(labels_dict):
    total = sum(labels_dict.values())
    max_num = max(labels_dict.values())
    mu = 1.0 / (total / max_num)
    class_weight = dict()
    for key, value in labels_dict.items():
        score = math.log(mu * total / float(value))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight
