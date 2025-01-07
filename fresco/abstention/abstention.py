"""
    Module implementing deep-abstaining classifier.

"""

#    pylint: disable=C0121

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn


class DacLoss(nn.Module):
    """
    Custom loss class for the DAC.

    Args:
        loss_fin: torch.nn.loss fun, either Cross Entropy or
            Binary Cross Entropy (for multilabel)
        multilabel: bool, multilabel problem?
        alphas: torch tensor of alphas for the DAC
    """

    def __init__(self, loss_fun, multilabel, alphas):
        super().__init__()
        self.loss_fun = loss_fun
        self.multilabel = multilabel
        self.alphas = alphas

    def forward(
        self,
        y_pred: torch.tensor,
        y_true: torch.tensor,
        idx: str,
        ntask_abs_prob: int = 1,
    ) -> torch.tensor:
        """Compute DAC loss.

        Params:
            y_pred - torch.tensor: logits from model.forward()
            y_true - torch.tensor: integer values of ground truth
            idx -  str:  task for alpha dict
            ntask_abs_prob: ntask abstaining score

        Pre-condition: loss function defined in class constructor

        """

        eps = 1e-4

        # updated loss
        if self.multilabel:
            log_probs = self._log_sigmoid(y_pred)
        else:  # need the log_softmax to account of correct operations for the multilabel case
            log_probs = F.log_softmax(y_pred, dim=1)

        p_k = log_probs[:, :-1]
        p_x = torch.exp(log_probs[:, -1])
        om_px = 1.0 - p_x + eps

        if self.multilabel:
            tmp_loss = torch.mean(self.loss_fun(p_k, y_true), -1) + torch.log(om_px)
        else:
            tmp_loss = self.loss_fun(p_k, y_true) + torch.log(om_px)

        loss = (om_px + 1 - ntask_abs_prob) * tmp_loss - self.alphas[idx] * torch.log(
            om_px
        )

        return torch.mean(loss)

    @staticmethod
    def _log_sigmoid(x):
        """Compute numerically stable log sigmoid function."""
        x_max = torch.amax(x, keep_dim=True)
        if not x_max.isfinite():
            x_max = 0.0
        tmp = x - x_max
        exp_tmp = torch.exp(tmp)
        s = torch.sum(exp_tmp, keepdim=True)
        out = torch.log(s)
        return tmp - out


class AbstainingClassifier:
    """Class for deep abastaining classifier.

    Params:
        id2label - dict : dict mapping int values to label values for each task
        kw_args - dict: dictionary with necessary keywords for abstention
        device - str: 'cuda' or 'cpu', set from model_suite.py
    """

    def __init__(
        self, kw_args: dict, device: torch.device, class_weights=None, clc: bool = False
    ):
        self.tasks = kw_args["data_kwargs"]["tasks"]
        self.ntask = kw_args["abstain_kwargs"]["ntask_flag"]

        if self.ntask:
            self.ntask_tasks = kw_args["abstain_kwargs"]["ntask_tasks"]
            self.ntask_alpha = kw_args["abstain_kwargs"]["ntask_alpha"]
            self.ntask_max_abs = kw_args["abstain_kwargs"]["ntask_max_abs"]
            self.ntask_min_acc = kw_args["abstain_kwargs"]["ntask_min_acc"]
            self.ntask_alpha_scale = kw_args["abstain_kwargs"]["ntask_alpha_scale"]

            self.ntask_min_scale = min(
                self.ntask_alpha_scale, 1.0 / self.ntask_alpha_scale
            )
            self.ntask_max_scale = max(
                self.ntask_alpha_scale, 1.0 / self.ntask_alpha_scale
            )
            self.ntask_filter = []
            self.ntask_acc = 0.0
            self.ntask_abs_rate = 0.0
            # self.ntask_acc_array = None  # this becomes a numpy array
        else:
            self.ntask_tasks = [None]

        # populated in self.add_abstention_classes
        self.n_classes = []
        self.abstain_labels = []

        self.accuracy = {task: [] for task in self.tasks}
        self.abs_rates = {}
        self.pred_idxs = {}

        # these need to be ordered according to self.tasks
        self.alphas = {
            task: float(kw_args["abstain_kwargs"]["alphas"][task])
            for task in self.tasks
        }

        self.max_abs = kw_args["abstain_kwargs"]["max_abs"]
        self.min_acc = kw_args["abstain_kwargs"]["min_acc"]
        self.alpha_scale = kw_args["abstain_kwargs"]["alpha_scale"]
        self.alpha_min_scale = {
            task: min(self.alpha_scale[task], 1.0 / self.alpha_scale[task])
            for task in self.tasks
        }
        self.alpha_max_scale = {
            task: max(self.alpha_scale[task], 1.0 / self.alpha_scale[task])
            for task in self.tasks
        }
        self.tune_mode = kw_args["abstain_kwargs"]["tune_mode"]
        self.abs_gain = kw_args["abstain_kwargs"]["abs_gain"]
        self.acc_gain = kw_args["abstain_kwargs"]["acc_gain"]
        self.stop_limit = kw_args["abstain_kwargs"]["stop_limit"]
        self.stop_metric = kw_args["abstain_kwargs"]["stop_metric"]

        if clc:
            self.base_loss = torch.nn.NLLLoss(reduction="none")
            self.multilabel = False
        else:
            if class_weights is not None:
                self.class_weights_tensor = torch.FloatTensor(
                    kw_args["train_kwargs"]["class_weights"]
                ).to(device)
            else:
                self.class_weights_tensor = None
            self.multilabel = kw_args["data_kwargs"]["multilabel"]
            if self.multilabel:
                self.base_loss = torch.nn.BCEWithLogitsLoss(
                    self.class_weights_tensor, reduction="none"
                )
            else:
                self.base_loss = torch.nn.NLLLoss(
                    self.class_weights_tensor, reduction="none"
                )

        self.dac_loss = DacLoss(self.base_loss, self.multilabel, self.alphas)

    def add_abstention_classes(self, dw):
        """Add abstention class, and ntask if enabled, to data wrangler attributes.

        Params:
            dw: dataHandler class

        Post-condition:
            dw.num_classes updated
            dw.dict_maps['id2label'] updated with abstention classes.
            self.abstain_labels populated
            self.n_classes populated.

        Must be called before creating PathReports class for DataLoaders.
        """
        self.abstain_labels = {
            task: len(dw.dict_maps["id2label"][task].keys()) for task in self.tasks
        }
        self.n_classes = {
            t: len(dw.dict_maps["id2label"][t].keys()) + 1 for t in self.tasks
        }
        dw.num_classes = copy.deepcopy(self.n_classes)

        for task in self.tasks:
            idx = len(dw.dict_maps["id2label"][task])
            if dw.dict_maps["id2label"][task][idx - 1] != f"abs_{task}":
                dw.dict_maps["id2label"][task][idx] = f"abs_{task}"

        if self.ntask:
            dw.num_classes["Ntask"] = 1

    def abstention_loss(
        self,
        y_pred: torch.tensor,
        y_true: torch.tensor,
        idx: int,
        ntask_abs_prob: float = 1,
    ) -> torch.tensor:
        """Compute DAC loss.

        Params:
            y_pred - torch.tensor: logits from model.forward()
            y_true - torch.tensor: integer values of ground truth
            idx: int index associated with a task
            ntask_abs_prob: probability of ntask abstaining on the document

        Pre-condition: loss function defined in class constructor

        """

        loss = self.dac_loss(y_pred, y_true, idx, ntask_abs_prob)
        return torch.mean(loss)

    def compute_accuracy(self, y_true, y_pred):
        """Compute abstain 0-1 accuracy and abstention rate for the DAC only.

        y_true and y_preds are dicts with keys as tasks and vals as lists of tensors.

        The attribute accuracy is cleared and repopulated within this function.

        """
        self.pred_idxs = {}
        self.abs_rates.clear()
        # clear accuracy scores from prior epoch

        for task in self.tasks:
            if task == "Ntask":
                continue
            # del self.accuracy[task][:]
            _preds = np.asarray([y.item() for y in y_pred[task]])
            # indices of abstained docs
            self.pred_idxs[task] = np.where(_preds != self.abstain_labels[task])[0]
            _trues = np.asarray(y_true[task])[self.pred_idxs[task]]
            if _preds.shape[0] > 0:
                self.abs_rates[f"{task}_abs"] = (
                    1.0 - _preds[self.pred_idxs[task]].shape[0] / _preds.shape[0]
                )
            else:
                self.abs_rates[f"{task}_abs"] = 1.0

            if _trues.shape[0] > 0 and _preds[self.pred_idxs[task]].shape[0] > 0:
                self.accuracy[task] = accuracy_score(_trues, _preds[self.pred_idxs[task]])
            else:
                self.accuracy[task] = 0.0

        return self.pred_idxs

    def get_ntask_filter(self, ntask_abs_prob):
        """Compute the ntask_mask.

        Note: self.ntask_filter is set to [] at the start of each epoch.

        called in training.compute_loss, must be called prior to calling compute_accuracy

        """
        # ntask 0-1, 0 - predict, 1 - abstain
        _filter = torch.round(ntask_abs_prob).type(torch.bool)
        # docs ntask is ok with, ie, ones to predict on
        self.ntask_filter.extend(torch.logical_not(_filter).tolist())

    def compute_ntask_accuracy(self, y_true: list, y_pred: list):
        """Compute ntask abstain 0-1 accuracy and abstention rate.

        y_true and y_preds are dicts with keys as tasks and vals as lists of tensors.

        """
        # need to get the length once for all tasks
        tmp = self.tasks[0]
        # array for Ntask accuracy over Ntask score
        ntask_preds = np.ones(len(y_true[tmp]))[self.ntask_filter]

        for task in self.ntask_tasks:
            _true = np.asarray(y_true[task])[self.ntask_filter]
            _pred = np.asarray([y.item() for y in y_pred[task]])[self.ntask_filter]
            if _true.shape[0] > 0 and _pred.shape[0] > 0:
                tmp_mask = _true == _pred
                ntask_preds = np.logical_and(ntask_preds, tmp_mask)

        if ntask_preds.shape[0] > 0:
            self.ntask_acc = (
                ntask_preds[ntask_preds == True].shape[0] / ntask_preds.shape[0]
            )
        else:
            self.ntask_acc = 0.0

        # this is the abstention rate based on only the Ntask score
        if isinstance(y_pred["Ntask"], list):

            ntask_pred = np.asarray(y_pred["Ntask"])
        else:
            ntask_pred = y_pred["Ntask"]

        self.ntask_abs_rate = (
            np.sum(ntask_pred) / ntask_pred.shape[0]
        )

        return (self.ntask_acc, self.ntask_abs_rate)

    def compute_abs_scores(self, y_true, y_pred, idx):
        """Compute accuracy scores for a dac model.

        Not presently used. 11/8/22

        """

        abs_scores = {}
        np_preds = np.asarray(y_pred)
        n_preds = np_preds[np_preds != self.abstain_labels[idx]]
        abs_rate = 1 - n_preds.shape[0] / len(y_pred)

        abs_scores["abs_rates"] = abs_rate
        abs_scores["abs_acc"] = accuracy_score(y_true, y_pred)
        abs_scores["alphas"] = self.alphas

        return abs_scores

    def modify_alphas(self, scores, additive=True):
        """Modify abstention alpha values.

        Params:
            scores: dict of dicts, keys are tasks, sub-dict key val pairs are
                micros - list: micro accruacy score
                abs_rate - list: abstention rate for each task

        Post-condition:
        self.alphas are modified in-place.

        """
        scale_factors = {task: None for task in self.tasks}
        stop_metrics = []

        for task in self.tasks:
            if task == "Ntask":
                continue
            # these are common to all tuning methods
            if scores[task]["micro"] == 0:
                # this works in the case of  0 predictions, but
                # not when there are 0 correct with N > 0 predictions
                # can get N from class attributes, # len(self.pred_idxs[task]) / self.abs_rate[task]
                acc_error = 1.0 - self.min_acc[task]
                acc_ratio = 1.0 / self.min_acc[task]
            else:
                acc_error = scores[task]["micro"] - self.min_acc[task]
                acc_ratio = scores[task]["micro"] / self.min_acc[task]
            abs_error = self.abs_rates[f"{task}_abs"] - self.max_abs[task]
            abs_ratio = self.abs_rates[f"{task}_abs"] / self.max_abs[task]

            if self.tune_mode == "abs_acc":
                # modify the scaling factor according to error in target abstention and accuracy

                # clip if accuracy is above min_acc
                acc_error = min([acc_error, 0.0])

                # clip if abstention is below max_abs
                abs_error = max([abs_error, 0.0])

                # multiplicative scaling
                # clip if accuracy is above min_acc
                acc_ratio = min([acc_ratio, 1.0])

                # clip if abstention is below max_abs
                abs_ratio = max([abs_ratio, 1.0])

                # choose multiplicative or additive scaling
                if additive:
                    new_scale = (
                        1.0 + self.acc_gain * acc_error + self.abs_gain * abs_error
                    )
                else:
                    new_scale = acc_ratio * abs_ratio

                # use harmonic mean to rescale the stopping criterion
                stop_i = (
                    (new_scale - 1.0)
                    * ((1.0 / self.acc_gain) + (1.0 / self.abs_gain))
                    * 0.5
                )

            elif self.tune_mode == "acc":
                new_scale = 1.0 + self.acc_gain * acc_error
                stop_i = acc_error
                # special case of there is no abstention and accuracy is more than requested
                # to avoid this task preventing stopping we set the stop metric to zero.
                if acc_error > 0.0 and abs_ratio < 1e-8:
                    new_scale = 1.0
                    stop_i = 0.0

            elif self.tune_mode == "abs":
                new_scale = 1.0 + self.abs_gain * abs_error
                stop_i = abs_error

            # threshold the scaling to be safe
            new_scale = min([new_scale, self.alpha_max_scale[task]])
            new_scale = max([new_scale, self.alpha_min_scale[task]])
            self.alphas[task] = self.alphas[task] * new_scale

            scale_factors[task] = new_scale
            stop_metrics.append(stop_i)

        return scale_factors, stop_metrics

    def modify_ntask_alpha(self, additive=True):
        """Modify ntask alpha value.

        Params:
            additive: bool, which type of scaling are we using?

        Post-condition:
            self.ntask_alpha modified in-place.

        """

        if self.ntask_acc == 0:
            acc_error = 1.0 - self.ntask_min_acc
            acc_ratio = 1.0 / self.ntask_min_acc
        else:
            acc_error = self.ntask_acc - self.ntask_min_acc
            acc_ratio = self.ntask_acc / self.ntask_min_acc
        abs_error = self.ntask_abs_rate - self.ntask_max_abs
        abs_ratio = self.ntask_abs_rate / self.ntask_max_abs

        if self.tune_mode == "abs_acc":
            # modify the scaling factor according to error in target abstention and accuracy

            # clip if accuracy is above min_acc
            acc_error = min([acc_error, 0.0])

            # clip if abstention is below max_abs
            abs_error = max([abs_error, 0.0])

            # multiplicative scaling
            # clip if accuracy is above min_acc
            acc_ratio = min([acc_ratio, 1.0])

            # clip if abstention is below max_abs
            abs_ratio = max([abs_ratio, 1.0])

            # choose multiplicative or additive scaling
            if additive:
                new_scale = 1.0 + self.acc_gain * acc_error + self.abs_gain * abs_error
            else:
                new_scale = acc_ratio * abs_ratio

            # use harmonic mean to rescale the stopping criterion
            stop_val = (
                (new_scale - 1.0)
                * ((1.0 / self.acc_gain) + (1.0 / self.abs_gain))
                * 0.5
            )

        elif self.tune_mode == "acc":
            new_scale = 1.0 + self.acc_gain * acc_error
            stop_val = acc_error

        elif self.tune_mode == "abs":
            new_scale = 1.0 + self.abs_gain * abs_error
            stop_val = abs_error

        # threshold the scaling to be safe
        new_scale = min([new_scale, self.ntask_max_scale])
        new_scale = max([new_scale, self.ntask_min_scale])
        self.ntask_alpha = self.ntask_alpha * new_scale

        return new_scale, stop_val

    def check_abs_stop_metric(self, stop_metrics):
        """Check if abstention stooping criteria is satisifed.

        The type of norm to use in the stopping critera is set the model_args file.

        """
        if self.stop_metric == "max":
            stop_val = np.linalg.norm(stop_metrics, np.inf)
        else:  # l2 norm
            stop_val = np.linalg.norm(stop_metrics)
        return stop_val

    @staticmethod
    def write_abs_header(tasks):
        """Write header for abstention stats output file."""
        path = "predictions/abs_stats.txt"

        with open(path, "w+", encoding="utf-8") as abs_file:
            abs_file.write("Alphas, accuracies, abstention, stop_metric\n")
            # we just want 4 copies on the header line
            for _ in range(4):
                for task in tasks:
                    abs_file.write(f"{task:10s} ")

            abs_file.write("\n")

    def write_abs_stats(self, stop_metrics):
        """Save abstention stats to output file."""
        path = "predictions/abs_stats.txt"
        with open(path, "a", encoding="utf-8") as abs_file:
            # write a single line with alphas, accuracy and abstention
            abs_file.write("Alphas:\n")
            for task in self.tasks:
                abs_file.write(f"{task:>9s}: {self.alphas[task]:10.5f} ")
            abs_file.write("\nDAC accuracy:\n")
            for task in self.tasks:
                abs_file.write(f"{task:>9s} {self.accuracy[task]:10.5f} ")
            abs_file.write("\nAbstention rates:\n")
            for k, v in self.abs_rates.items():
                abs_file.write(f"{k:>9s}: {v:10.5f} ")
            abs_file.write("\nStopping values:\n")
            for i, task in enumerate(self.tasks):
                abs_file.write(f"{task:>9s}: {stop_metrics[i]:10.5f} ")
            abs_file.write("\n")

    def print_abs_tune_header(self):
        """Change output header based on tuning mode."""
        if self.tune_mode == "abs_acc":
            print(
                (f"{'task':12s}: {'macro':>10s}, {'micro':>10s}, {'min_acc':>10s},  ")
                + (f"{'abs_frac':>10s}, {'max_abs':>10s}, {'alpha':>9s}, ")
                + (f"{'scale_frac':>12s}, {'stop_metric':>12s}")
            )
        elif self.tune_mode == "abs":
            print(
                (f"{'task':12s}: {'macro':>10s}, {'micro':>10s}, ")
                + (f"{'abs_frac':>10s}, {'max_abs':>10s}, {'alpha':>9s}, ")
                + (f"{'scale_frac':>12s}, {'stop_metric':>12s}")
            )
        elif self.tune_mode == "acc":
            print(
                (f"{'task':12s}: {'macro':>10s}, {'micro':>10s},  ")
                + (f"{'abs_frac':>10s}, {'target_acc':>10s}, {'alpha':>9s}, ")
                + (f"{'scale_frac':>12s}, {'stop_metric':>12s}")
            )

    def print_abs_tune_stats(
        self,
        task,
        macro,
        micro,
        min_acc,
        abs_frac,
        max_abs,
        alpha,
        scale_frac,
        stop_metric,
    ):
        """Print output based on tuning mode."""
        if self.tune_mode == "abs_acc":
            print(
                (f"{task:12s}: {macro:10.4f}, {micro:10.4f}, {min_acc:10.4f}, ")
                + (f"{abs_frac:10.4f}, {max_abs:10.4f}, {alpha:10.4f},")
                + (f"{scale_frac:10.4f}, {stop_metric:10.4f}")
            )
        elif self.tune_mode == "abs":
            print(
                (
                    f"{task:12s}: {macro:10.4f}, {micro:10.4f}, {abs_frac:10.4f}, "
                    + f"{max_abs:10.4f}, {alpha:10.4f}, {scale_frac:10.4f}, {stop_metric:10.4f}"
                )
            )
        elif self.tune_mode == "acc":
            print(
                (
                    f"{task:12s}: {macro:10.4f}, {micro:10.4f}, {abs_frac:10.4f}, "
                    + f"{min_acc:10.4f}, {alpha:10.4f}, {scale_frac:10.4f}, {stop_metric:10.4f}"
                )
            )

    @staticmethod
    def print_abs_header():
        """Set format output header line."""
        print(f"{'task':12s}: {'macro':>10s}, {'micro':>10s}, {'abs_frac':>10s}")

    @staticmethod
    def print_abs_stats(task, micro, macro, abs_frac):
        """Pring abstention statis diring training."""
        print(f"{task:12s}: {macro:10.4f}, {micro:10.4f}, {abs_frac:10.4f}")
