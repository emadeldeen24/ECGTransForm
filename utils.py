import torch
import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, exp_log_dir, seed_id):
    log_dir = os.path.join(exp_log_dir, "_seed_" + str(seed_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug("=" * 45)
    logger.debug(f'Seed: {seed_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(exp_log_dir, model, dataset, dataset_configs, hparams, status):
    save_dict = {
        "dataset": dataset,
        "configs": dataset_configs.__dict__,
        "hparams": dict(hparams),
        "model": model.state_dict()
    }
    # save classification report
    save_path = os.path.join(exp_log_dir, f"checkpoint_{status}.pt")

    torch.save(save_dict, save_path)


def _calc_metrics(pred_labels, true_labels, classes_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=classes_names, digits=6, output_dict=True)
    accuracy = accuracy_score(true_labels, pred_labels)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


def _save_metrics(pred_labels, true_labels, log_dir, status):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = f"classification_report_{status}.xlsx"
    report_Save_path = os.path.join(log_dir, file_name)
    df.to_excel(report_Save_path)


import collections


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.abc.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models.py", os.path.join(destination_dir, f"models.py"))
    copy(f"configs/data_configs.py", os.path.join(destination_dir, f"data_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"trainer.py", os.path.join(destination_dir, f"trainer.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))


def _plot_umap(model, data_loader, device, save_dir):
    import umap
    import umap.plot
    from matplotlib.colors import ListedColormap
    classes_names = ['N','S','V','F','Q']
    
    font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 17}
    plt.rc('font', **font)
    
    with torch.no_grad():
        # Source flow
        data = data_loader.dataset.x_data.float().to(device)
        labels = data_loader.dataset.y_data.view((-1)).long()
        out = model[0](data)
        features = model[1](out)


    if not os.path.exists(os.path.join(save_dir, "umap_plots")):
        os.mkdir(os.path.join(save_dir, "umap_plots"))
        
    #cmaps = plt.get_cmap('jet')
    model_reducer = umap.UMAP() #n_neighbors=3, min_dist=0.3, metric='correlation', random_state=42)
    embedding = model_reducer.fit_transform(features.detach().cpu().numpy())
    
    # Normalize the labels to [0, 1] for colormap
    norm_labels = labels / 4.0
    

    # Create a new colormap by extracting the first 5 colors from "Paired"
    paired = plt.cm.get_cmap('Paired', 12)  # 12 distinct colors
    new_colors = [paired(0), paired(1), paired(2), paired(4), paired(6)]  # Skip every second color, but take both from the first pair
    new_cmap = ListedColormap(new_colors)

    print("Plotting UMAP ...")
    plt.figure(figsize=(16, 10))
    # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels,  s=10, cmap='Spectral')
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=norm_labels, cmap=new_cmap, s=15)

    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, classes_names,  title="Classes")
    file_name = "umap_.png"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()
