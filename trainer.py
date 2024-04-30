import torch
import torch.nn.functional as F
import datetime
import os
import collections
import numpy as np

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from models import ecgTransForm
from dataloader import data_generator
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import AverageMeter, to_device, _save_metrics, copy_Files
from utils import fix_randomness, starting_logs, save_checkpoint, _calc_metrics


class trainer(object):
    def __init__(self, args):
        # dataset parameters
        self.dataset = args.dataset
        self.seed_id = args.seed_id

        self.device = torch.device(args.device)

        # Exp Description
        self.run_description = f"{args.run_description}_{datetime.datetime.now().strftime('%H_%M')}"
        self.experiment_description = args.experiment_description

        # paths
        self.home_path = os.getcwd()
        self.save_dir = os.path.join(os.getcwd(), "experiments_logs")
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, self.run_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        self.data_path = args.data_path


        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # Specify hparams
        self.hparams = self.hparams_class.train_params

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class("supervised")
        return dataset_class(), hparams_class()

    def load_data(self, data_type):
        self.train_dl, self.val_dl, self.test_dl, self.cw_dict = \
            data_generator(self.data_path, data_type, self.hparams)

    def calc_results_per_run(self):
        acc, f1 = _calc_metrics(self.pred_labels, self.true_labels, self.dataset_configs.class_names)
        return acc, f1

    def train(self):
        copy_Files(self.exp_log_dir)  # save a copy of training files

        self.metrics = {'accuracy': [], 'f1_score': []}

        # fixing random seed
        fix_randomness(int(self.seed_id))

        # Logging
        self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.exp_log_dir, self.seed_id)
        self.logger.debug(self.hparams)
        
        # Load data
        self.load_data(self.dataset)

        model = ecgTransForm(configs=self.dataset_configs, hparams=self.hparams)
        model.to(self.device)

        # Average meters
        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            betas=(0.9, 0.99)
        )
        
        weights = [float(value) for value in self.cw_dict.values()]
        # Now convert the list of floats to a numpy array, then to a PyTorch tensor
        weights_array = np.array(weights).astype(np.float32)  # Ensuring the correct dtype
        weights_tensor = torch.tensor(weights_array).to(self.device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        best_acc = 0
        best_f1 = 0

        # training..
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            model.train()

            for step, batches in enumerate(self.train_dl):
                batches = to_device(batches, self.device)

                data = batches['samples'].float()
                labels = batches['labels'].long()

                # ====== Source =====================
                self.optimizer.zero_grad()

                # Src original features
                logits = model(data)

                # Cross-Entropy loss
                x_ent_loss = self.cross_entropy(logits, labels)

                x_ent_loss.backward()
                self.optimizer.step()

                losses = {'Total_loss': x_ent_loss.item()}
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, self.hparams["batch_size"])

            self.evaluate(model, self.val_dl)
            tr_acc, tr_f1 = self.calc_results_per_run()
            # logging
            self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in loss_avg_meters.items():
                self.logger.debug(f'{key}\t: {val.avg:2.4f}')
            self.logger.debug(f'TRAIN: Acc:{tr_acc:2.4f} \t F1:{tr_f1:2.4f}')

            # VALIDATION part
            self.evaluate(model, self.val_dl)
            ts_acc, ts_f1 = self.calc_results_per_run()
            if ts_f1 > best_f1:  # save best model based on best f1.
                best_f1 = ts_f1
                best_acc = ts_acc
                save_checkpoint(self.exp_log_dir, model, self.dataset, self.dataset_configs, self.hparams, "best")
                _save_metrics(self.pred_labels, self.true_labels, self.exp_log_dir, "validation_best")

            # logging
            self.logger.debug(f'VAL  : Acc:{ts_acc:2.4f} \t F1:{ts_f1:2.4f} (best: {best_f1:2.4f})')
            self.logger.debug(f'-------------------------------------')

            # LAST EPOCH
        _save_metrics(self.pred_labels, self.true_labels, self.exp_log_dir, "validation_last")
        self.logger.debug("LAST EPOCH PERFORMANCE on validation set...")
        self.logger.debug(f'Acc:{ts_acc:2.4f} \t F1:{ts_f1:2.4f}')

        self.logger.debug(":::::::::::::")
        # BEST EPOCH
        self.logger.debug("BEST EPOCH PERFORMANCE on validation set ...")
        self.logger.debug(f'Acc:{best_acc:2.4f} \t F1:{best_f1:2.4f}')
        save_checkpoint(self.exp_log_dir, model, self.dataset, self.dataset_configs, self.hparams, "last")


        # TESTING
        print(" === Evaluating on TEST set ===")
        self.evaluate(model, self.test_dl)
        test_acc, test_f1 = self.calc_results_per_run()
        _save_metrics(self.pred_labels, self.true_labels, self.exp_log_dir, "test_last")
        self.logger.debug(f'Acc:{test_acc:2.4f} \t F1:{test_f1:2.4f}')


    def evaluate(self, model, dataset):
        model.to(self.device).eval()

        total_loss_ = []

        self.pred_labels = np.array([])
        self.true_labels = np.array([])

        with torch.no_grad():
            for batches in dataset:
                batches = to_device(batches, self.device)
                data = batches['samples'].float()
                labels = batches['labels'].long()

                # forward pass
                predictions = model(data)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.pred_labels = np.append(self.pred_labels, pred.cpu().numpy())
                self.true_labels = np.append(self.true_labels, labels.data.cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()  # average loss
