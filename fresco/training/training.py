"""
    Module for training a deep learning model>
"""
import copy
import datetime
import os
import pickle
import sys
import time

import torch
# import torch.nn.functional as F

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from torch.profiler import profile, record_function, ProfilerActivity, schedule


def trace_handler(p):
    output = p.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=50)
    print(output)
    p.export_chrome_trace(f"trace_{p.step_num}.json")


class ModelTrainer():
    """
    Training class definition.

    Attributes:
        savepath (str): Path for saving models and metrics.
        epochs (int): Maximum number of epochs to train for.
        patience_stop (int): Patience stopping criteria.
        tasks (list): List of tasks, each task is a string.
        n_task (bool): Are we using ntask?

        model (Model): Model definition, declared and initialized in the caller.
        dw (DataHandler): DataHandler class, initialized in the caller.
        device (torch.device): CUDA or CPU.
        best_loss (float): Best validation loss scores.
        patience_ctr (int): Patience counter.
        loss (torch.tensor): Loss value on device.

        y_preds (dict): Dict of predictions, usually logits as torch.tensor, tasks are key values
        y_trues (dict): Dict of ints with ground truth values, tasks are key values.

        multilabel (bool): Multilabel classification?
        abstain (bool): Use the deep abstaining classifier?
        mixed_precision (bool): Use PyTorch automatic mixed precision?

        opt (torch.optimizer): Optimizer for training.
        reduction (str): Type of reduction for the loss function.

        loss_funs (dict): Dictionary of torch loss functions for training each task.

        class_weights (list): List of floats for class weighting schemes.
    """
    def __init__(self, kw_args, model, dw, class_weights=None, device=None, fold=None, clc=False):

        path = 'savedmodels/' + kw_args['save_name'] + "/"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path))

        if fold is None:
            self.fold = kw_args['data_kwargs']['fold_number']
        else:
            self.fold = fold

        self.savepath = path
        self.savename = path + kw_args['save_name'] + f"_fold{self.fold}.h5"
        self.class_weights = class_weights

        self.abstain = kw_args['abstain_kwargs']['abstain_flag']
        self.ntask = kw_args['abstain_kwargs']['ntask_flag']
        self.mixed_precision = kw_args['train_kwargs']['mixed_precision']

        # setup loss function
        if self.abstain:
            reduction = 'none'
        else:
            reduction = 'mean'

        self.clc = clc

        self.tasks = kw_args['data_kwargs']['tasks']

        self.model = model
        self.device = device
        self.best_loss = np.inf

        if self.clc:
            self.loss_fun = torch.nn.CrossEntropyLoss(self.class_weights, reduction=reduction)
        else:
            self.multilabel = kw_args['train_kwargs']['multilabel']
            # setup class weights
            if kw_args['train_kwargs']['class_weights'] is not None:
                with open(kw_args['train_kwargs']['class_weights'],'rb') as f:
                    weights = pickle.load(f)
                self.loss_funs = {}
                for task in self.tasks:
                    weights_task = torch.FloatTensor(weights[task]).to(self.device, non_blocking=True)
                    if self.multilabel:
                        self.loss_funs[task] = torch.nn.BCEWithLogitsLoss(weights_task,
                                reduction=reduction)
                    else:
                        self.loss_funs[task] = torch.nn.CrossEntropyLoss(weights_task,
                                reduction=reduction)
            else:
                self.loss_funs = {}
                for task in self.tasks:
                    if self.multilabel:
                        self.loss_funs[task] = torch.nn.BCEWithLogitsLoss(None, reduction=reduction)
                    else:
                        self.loss_funs[task] = torch.nn.CrossEntropyLoss(None, reduction=reduction)

        self.patience_ctr = 0
        self.epochs = kw_args['train_kwargs']['max_epochs']
        self.patience_stop = kw_args['train_kwargs']['patience']

        self.bs = kw_args['train_kwargs']['batch_per_gpu']

        if self.clc:
            self.y_preds = {task: [] for task in self.tasks}
            self.y_trues = {task: [] for task in self.tasks}
            self.val_preds = {task: [] for task in self.tasks}
            self.val_trues = {task: [] for task in self.tasks}
        else:
            self.y_preds = {task: np.empty((dw.train_size)) for task in self.tasks}
            self.y_trues = {task: np.empty((dw.train_size)) for task in self.tasks}
            if dw.val_size > 0:
                val_size = dw.val_size
            else:  # train is used for val
                val_size = dw.train_size
            self.val_preds = {task: np.empty((val_size)) for task in self.tasks}
            self.val_trues = {task: np.empty((val_size)) for task in self.tasks}

        self.opt = torch.optim.Adam(self.model.parameters(), 0.0001, (0.9, 0.99))

    def get_ys(self, logits, y, idx, val=False):
        """
        Get ground truth and y_predictions.

        Args:
            logits (list): List of logits.
            y (dict): Dictionary of numpy ndarrays, with tasks as keys.
            idx (int): Index in the enumerated DataLoader.

        Post-condition:
            self.y_trues and self.y_preds are populated.
        """

        if val:
            preds = self.val_preds
            trues = self.val_trues
        else:
            preds = self.y_preds
            trues = self.y_trues

        for i, task in enumerate(self.tasks):
            if logits[i].shape[0] == self.bs:
                preds[task][idx*self.bs:(idx+1)*self.bs] = np.argmax(logits[i].detach().cpu().numpy(), 1)
                if self.multilabel:
                    trues[task][idx*self.bs:(idx+1)*self.bs] = np.argmax(y[task].detach().cpu().numpy(), 1)
                else:
                    trues[task][idx*self.bs:(idx+1)*self.bs] = y[task].detach().cpu().numpy()
            else:
                preds[task][idx*self.bs:] = np.argmax(logits[i].detach().cpu().numpy(), 1)
                if self.multilabel:
                    trues[task][idx*self.bs:] = np.argmax(y[task].detach().cpu().numpy(), 1)
                else:
                    trues[task][idx*self.bs:] = y[task].detach().cpu().numpy()

    def profile_fit_model(self, train_loader, dac=None):
        """
        Main training loop.

        Args:
            train_loader (torch.DataLoader): Initialized and populated in the calling function.
            dac (Abstention): Abstention class, deep abstaining classifier class.
        """

        for epoch in range(self.epochs):
            print(f'\nepoch: {epoch+1}', flush=True)
            self.model.train()
            if self.ntask:
                dac.ntask_filter = []
            if self.clc:
                for task in self.tasks:
                    del self.y_preds[task][:]
                    del self.y_trues[task][:]

            start_time = time.time()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=5),
                on_trace_ready=trace_handler
            ) as p:
                for i, batch in enumerate(train_loader):
                    self.opt.zero_grad()
                    X = batch["X"].to(self.device, non_blocking=True)
                    y = {task: batch[f"y_{task}"].to(self.device, non_blocking=True)
                         for task in self.tasks}
                    if self.clc:
                        batch_len = batch['len'].to(self.device, non_blocking=True)
                        max_seq_len = X.shape[1]
                        mask = torch.arange(end=max_seq_len, device=self.device)[None, :] < batch_len[:, None]
                        idxs = torch.nonzero(mask, as_tuple=True)
                        logits = self.model(X, batch_len)
                        loss = self.compute_clc_loss(logits, y, idxs, dac)
                    else:
                        logits = self.model(X)
                        loss = self.compute_loss(logits, y, dac)
                        self.get_ys(logits, y, i)

                    # backprop
                    loss.backward()
                    self.opt.step()
                    p.step()

            print(f"\ntraining time {time.time() - start_time:.2f}", flush=True)
            sys.stdout.flush()

            loss_np = loss.detach().cpu().item()
            print(f"Training loss: {loss_np:.6f}")

    def fit_model(self, train_loader, val_loader=None, dac=None):
        """
        Main training loop.

        Args:
            train_loader (torch.DataLoader): Initialized and populated in the calling function.
            val_loader (torch.DataLoader): Initialized and populated in the calling function. Can be None, then training is used for validation scores.
            dac (Abstention): Abstention class, deep abstaining classifier class.
        """

        all_scores = {}
        best_loss = np.inf
        model_weights = None
        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        for epoch in range(self.epochs):
            print(f'\nepoch: {epoch+1}', flush=True)
            self.model.train()
            if self.ntask:
                dac.ntask_filter = []
            if self.clc:
                for task in self.tasks:
                    del self.y_preds[task][:]
                    del self.y_trues[task][:]

            start_time = time.time()

            for i, batch in enumerate(train_loader):
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    self.opt.zero_grad()
                    X = batch["X"].to(self.device, non_blocking=True)
                    y = {task: batch[f"y_{task}"].to(self.device, non_blocking=True)
                         for task in self.tasks}
                    if self.clc:
                        batch_len = batch['len'].to(self.device, non_blocking=True)
                        max_seq_len = X.shape[1]
                        mask = torch.arange(end=max_seq_len, device=self.device)[None, :] < batch_len[:, None]
                        idxs = torch.nonzero(mask, as_tuple=True)
                        logits = self.model(X, batch_len)
                        loss = self.compute_clc_loss(logits, y, idxs, dac)
                    else:
                        logits = self.model(X)
                        loss = self.compute_loss(logits, y, dac)
                        self.get_ys(logits, y, i)

                # backprop
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()

            print(f"\ntraining time {time.time() - start_time:.2f}", flush=True)
            sys.stdout.flush()

            loss_np = loss.detach().cpu().item()
            print(f"Training loss: {loss_np:.6f}")

            train_scores = self.train_metrics(dac)
            train_scores['train_loss'] = loss_np
            all_scores[f'epoch_{epoch}_train_scores'] = train_scores

            print(f"\nepoch {epoch+1} validation\n", flush=True)
            stop, val_scores = self.score(epoch, val_loader=val_loader, dac=dac)
            all_scores[f'epoch_{epoch}_val_scores'] = val_scores

            if val_scores['val_loss'][0] < best_loss:
                best_loss = val_scores['val_loss'][0]
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'opt_state_dict': self.opt.state_dict(),
                            'val_loss': best_loss
                            }, self.savename)

            if stop:
                print(f"saving to {self.savename}", flush=True)
                # loading weights of best model
                checkpoint = torch.load(self.savename)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                scores = f"epoch_{epoch}_scores_fold{self.fold}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
                with open(self.savepath + scores, "wb") as f_out:
                    pickle.dump(all_scores, f_out, pickle.HIGHEST_PROTOCOL)
                break
        if epoch + 1 == self.epochs:
            print('\nModel training hit max epochs, not converged')
            # loading weights of best model
            checkpoint = torch.load(self.savename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"saving to {self.savename}", flush=True)
            scores = f"epoch_{epoch}_scores_fold{self.fold}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
            with open(self.savepath + scores, "wb") as f_out:
                pickle.dump(all_scores, f_out, pickle.HIGHEST_PROTOCOL)

    def train_metrics(self, dac=None):
        """
        Compute per-epoch metrics during training.
        """

        scores = {task: {} for task in self.tasks}

        if self.abstain:
            pred_idxs = dac.compute_accuracy(self.y_trues, self.y_preds)
            if self.ntask:
                ntask_scores = dac.compute_ntask_accuracy(self.y_trues, self.y_preds)
                scores['ntask'] = {}
                scores['ntask']['ntask_acc'] = ntask_scores[0]
                scores['ntask']['ntask_abs_rate'] = ntask_scores[1]

        for task in self.tasks:
            _trues = self.y_trues[task]
            _preds = self.y_preds[task]

            if self.abstain:
                _true = np.array(_trues)
                _pred = np.array(_preds)
                scores[task]['macro'] = f1_score(_true[pred_idxs[task]],
                                                 _pred[pred_idxs[task]],
                                                 average='macro')
                scores[task]['micro'] = f1_score(_true[pred_idxs[task]],
                                                 _pred[pred_idxs[task]],
                                                 average='micro')
                scores[task]['abs_rate'] = dac.abs_rates[f'{task}_abs']
            else:
                scores[task]['macro'] = f1_score(_trues, _preds, average='macro')
                scores[task]['micro'] = f1_score(_trues, _preds, average='micro')

        # print/write stats
        if self.abstain:
            dac.print_abs_header()
            for task in self.tasks:
                dac.print_abs_stats(task, scores[task]['micro'],
                                    scores[task]['macro'],
                                    scores[task]['abs_rate'])
        else:
            print(f"{'task':>12s}: {'micro':>10s} {'macro':>12s}")
            for task in self.tasks:
                print(f"{task:>12s}: {scores[task]['micro']:>10.4f}, {scores[task]['macro']:>10.4f}")

        if self.ntask:
            print(f"{'ntask':12s}: {ntask_scores[0]:10.4f}, " +
                  f"{ntask_scores[0]:10.4f}, {ntask_scores[1]:10.4f}")

        return scores

    def score(self, epoch, val_loader=None, dac=None):
        """
        Score a model during training.

        Args:
            epoch (int): Epoch number.
            val_loader (torch.dataLoader): DataLoader class with PathReports.
            dac (Abstention): Deep abstaining classifier class. Are we using the dac?
        """

        val_scores = {'val_loss': [], 'abs_stop_vals': [],
                      'val_micro': [], 'val_macro': []}
        if self.abstain:
            abs_scores = {}
            # abs_stop_vals = []

        if self.clc:
            for task in self.tasks:
                del self.val_preds[task][:]
                del self.val_trues[task][:]

        if val_loader is not None:
            scores = self._score(data_loader=val_loader, dac=dac)
        else:  # score the training set
            self.val_trues = self.y_trues
            self.val_preds = self.y_preds

        if self.abstain:
            pred_idxs = dac.compute_accuracy(self.val_trues, self.val_preds)
            if self.ntask:
                ntask_scores = dac.compute_ntask_accuracy(self.val_trues, self.val_preds)
                scores['ntask'] = {}
                scores['ntask']['ntask_acc'] = ntask_scores[0]
                scores['ntask']['ntask_abs_rate'] = ntask_scores[1]

        for task in self.tasks:
            _trues = self.val_trues[task]
            _preds = self.val_preds[task]
            if self.abstain:
                score, _abs_scores = self.compute_scores(np.array(_trues)[pred_idxs[task]],
                                                         np.array(_preds)[pred_idxs[task]],
                                                         task, dac)
                abs_scores[task] = _abs_scores
            else:
                score, _ = self.compute_scores(_trues, _preds, task)

            scores[task] = score

        if self.abstain:
            alpha_scale, abs_stop_vals = dac.modify_alphas(abs_scores)
            if self.ntask:
                ntask_scale, ntask_stop_val = dac.modify_ntask_alpha()
                abs_stop_vals.append(ntask_stop_val)

            stop, _ = self.stop_metrics(scores['val_loss'],
                                        epoch,
                                        abs_stop_vals,
                                        dac)
            val_scores['abs_stop_vals'].append(abs_stop_vals)
        else:
            stop, _ = self.stop_metrics(scores['val_loss'], epoch)

        val_scores['val_loss'].append(scores['val_loss'])
        macro = {task: scores[task]['macro'] for task in self.tasks}
        micro = {task: scores[task]['micro'] for task in self.tasks}
        val_scores['val_macro'].append(macro)
        val_scores['val_micro'].append(micro)

        # print/write stats
        if self.ntask:
            self.output_scores(scores, abs_scores=abs_scores,
                               dac=dac, stop_vals=abs_stop_vals, alpha_scale=alpha_scale,
                               ntask_stop_val=ntask_stop_val, ntask_alpha_scale=ntask_scale)
        elif self.abstain:
            self.output_scores(scores, abs_scores, dac, abs_stop_vals, alpha_scale)
        else:
            self.output_scores(scores)

        if not stop and self.abstain:
            # update alphas
            val_scores['abs_stop_vals'].append(abs_stop_vals)
            if self.ntask:
                print(f"Updated ntask alpha: {dac.ntask_alpha:0.6f}")
            print('Updated alphas: ', dac.alphas)

        return stop, val_scores

    def _score(self, data_loader=None, dac=None):
        """
        Score data_loader for validation.

        Args:
            data_loader (torch.DataLoader): Typically the validation split. If None, then use the training set.
            dac (Abstention): Abstention class, deep abstaining classifier class.

        Post condition:
            self.val_preds and self.val_trues are updated.

        Returns:
            scores (dict): Dictionary where keys are tasks, and values are dictionaries with keys val_loss, macro, and micro.
            abs_scores (list): List of abstention scores.
        """
        scores = {}
        losses = np.empty(len(data_loader))
        self.model.eval()

        if self.ntask:
            dac.ntask_filter = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    X = batch["X"].to(self.device, non_blocking=True)
                    y = {task: batch[f"y_{task}"].to(self.device, non_blocking=True)
                         for task in self.tasks}
                    if self.clc:
                        batch_len = batch['len'].to(self.device, non_blocking=True)
                        max_seq_len = X.shape[1]
                        mask = torch.arange(end=max_seq_len, device=self.device)[None, :] < batch_len[:, None]
                        idxs = torch.nonzero(mask, as_tuple=True)
                        logits = self.model(X, batch_len)
                        losses[i] = self.compute_clc_loss(logits, y, idxs, dac, val=True).detach().cpu().numpy()
                    else:
                        logits = self.model(X)
                        losses[i] = self.compute_loss(logits, y, dac).detach().cpu().numpy()
                        self.get_ys(logits, y, i, val=True)
        scores['val_loss'] = np.mean(losses)

        return scores

    def compute_scores(self,
                       y_true,
                       y_pred,
                       task,
                       dac=None):
        """
        Compute macro/micro scores per task.

        Args:
            y_true (list): List of ground truth labels (as int). It is a list of lists for n_tasks > 1.
            y_pred (list): List of predicted classes (as list of tensors). It is a list of lists for n_tasks > 1.
            dac (Abstention): Deep abstaining classifier class.
            idx (int): Indexing self.tasks.

        Returns:
            scores:
        """
        scores = {}
        _y_pred = [y.item() for y in y_pred]
        micro = f1_score(y_true, _y_pred, average='micro')
        scores['micro'] = micro
        macro = f1_score(y_true, _y_pred, average='macro')
        scores['macro'] = macro

        if self.abstain:
            abs_scores = {}  # dac.compute_abs_scores(y_true, _y_pred, tasks)
            abs_scores['macro'] = macro
            abs_scores['micro'] = micro
            abs_scores['stop_metrics'] = macro
            abs_scores['abs_rates'] = dac.abs_rates[f"{task}_abs"]
            abs_scores['abs_acc'] = accuracy_score(y_true, _y_pred)
        else:
            abs_scores = None

        return scores, abs_scores

    def output_scores(self,
                      scores,
                      abs_scores=None,
                      dac=None,
                      stop_vals=None,
                      alpha_scale=None,
                      ntask_stop_val=None,
                      ntask_alpha_scale=None):
        """
        Print stats to the terminal.

        Args:
            scores (dict): Dictionary of metrics, with tasks as keys and metrics as values.
            dac (Abstaining): Abstaining Classifier class.
            stop_vals (list): List of stopping criteria.
            alpha_scale (dict): Dictionary of scaling values for the dac, with tasks as keys.
            ntask_stop_val (float): Stopping criterion for ntask.
            ntask_alpha_scale (float): Scaling factor for ntask alpha.

        """
        if self.abstain:
            abs_micros = [abs_scores[task]['micro'] for task in self.tasks]
            dac.write_abs_stats(abs_micros)

            dac.print_abs_tune_header()
            for i, task in enumerate(self.tasks):
                dac.print_abs_tune_stats(task, scores[task]['macro'],
                                         scores[task]['micro'], dac.min_acc[task],
                                         abs_scores[task]['abs_rates'], dac.max_abs[task],
                                         dac.alphas[i], alpha_scale[task],
                                         stop_vals[i])
            # print ntask stats
            if self.ntask:
                dac.print_abs_tune_stats('ntask', dac.ntask_acc, dac.ntask_acc,
                                         dac.ntask_min_acc, dac.ntask_abs_rate,
                                         dac.ntask_max_abs, dac.ntask_alpha,
                                         ntask_alpha_scale, ntask_stop_val)
        else:
            print(f"{'task':>12s}: {'micro':>10s} {'macro':>12s}")
            self.print_stats(scores)

    def compute_loss(self, logits, y, dac=None):
        """
        Compute forward pass and loss function.

        Args:
            logits (torch.tensor): Logits.
            dac (Abstention): Deep abstaining classifier class.
            ntask_abs (float): Probability of abstaining on the entire document.

        Returns:
            loss (torch.tensor): Float tensor.

        Post-condition:
            y_preds and y_trues are populated.
            loss is updated.
        """

        loss = 0.0
        if self.ntask:
            ntask_abs = torch.sigmoid(logits[-1])[:, -1]
            dac.get_ntask_filter(ntask_abs)
        for i, task in enumerate(self.tasks):
            if self.ntask:
                if task in dac.ntask_tasks:
                    loss += dac.abstention_loss(logits[i], y[task], i, ntask_abs_prob=ntask_abs)
                else:
                    loss += dac.abstention_loss(logits[i], y[task], i)
            elif self.abstain:  # just the dac
                loss += dac.abstention_loss(logits[i], y[task], i)
            else:  # nothing fancy
                loss += self.loss_funs[task](logits[i], y[task])

        if self.ntask:
            loss = loss - torch.mean(dac.ntask_alpha * torch.log(1 - ntask_abs + 1e-6))
        # average over all tasks
        return loss / len(self.tasks)

    def compute_clc_loss(self, logits, y, idxs, dac=None, val=False):
        """
        Compute forward pass and case level loss function.

        Args:
            batch (torch.tensor): Iterate from DataLoader.
            dac (Abstention): Deep abstaining classifier class.
            ntask_abs (float): Probability of abstaining on the entire document.

        Returns:
            loss (torch.tensor): Float tensor.

        Post-condition:
            y_preds and y_trues are populated.
            loss is updated.
        """

        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if val:
            y_trues = self.val_trues
            y_preds = self.val_preds
        else:
            y_trues = self.y_trues
            y_preds = self.y_preds

        if self.ntask:
            ntask_abs = torch.sigmoid(logits[-1][idxs])[:, -1]
            dac.get_ntask_filter(ntask_abs)

        for i, task in enumerate(self.tasks):
            y_true = y[task][idxs]
            y_pred = logits[i][idxs]

            if self.ntask:
                if task in dac.ntask_tasks:
                    loss += dac.abstention_loss(y_pred, y_true, i, ntask_abs_prob=ntask_abs)
                else:
                    loss += dac.abstention_loss(y_pred, y_true, i)
            elif self.abstain:  # just the dac
                loss += dac.abstention_loss(y_pred, y_true, i)
            else:  # nothing fancy
                loss += self.loss_fun(y_pred, y_true)
            y_preds[task].extend(np.argmax(y_pred.detach().cpu().numpy(), 1))
            y_trues[task].extend(y_true.detach().cpu().numpy())

        if self.ntask:
            loss = loss - torch.mean(dac.ntask_alpha * torch.log(1 - ntask_abs + 1e-6))
        # average over all tasks
        return loss / len(self.tasks)

    def stop_metrics(self, loss, epoch, stop_metrics=None, dac=None):
        """
        Compute stop metrics.

        Compute stop metrics for normal (val loss + patience) or DAC (stop metric) training.

        Args:
            loss (torch.tensor): Float tensor, val loss value at the current epoch.
            epoch (int): Epoch counter.
            stop_metrics (dict): Dictionary of floats for abstention rates and accuracy.

        Returns:
            If abstaining:
                stop_val (float): DAC stopping criterion.
                stop (bool): Stop or go?
            Otherwise:
                stop_val (float): Best val loss.
                stop (bool): Stop or go?

        Post-condition:
        If not abstaining, patience counter is updated.
        """

        if self.abstain:
            stop_val = dac.check_abs_stop_metric(np.asarray(stop_metrics))
            if stop_val < dac.stop_limit:
                print(f'Stopping criterion reached: {stop_val:.4f} < {dac.stop_limit:.4f}')
                stop = True
            else:
                print(f'Stopping criterion not reached: {stop_val:.4f} > {dac.stop_limit:.4f}')
                stop = False
        else:
            stop_val = None
            stop = False
            print(f"epoch {epoch+1:d} val loss: {loss:.8f}, best val loss: {self.best_loss:.8f}")
            # use patience based on val loss
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_ctr = 0
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.patience_stop:
                    stop = True
            print(f'patience counter is at {self.patience_ctr} of {self.patience_stop}')
        return stop, stop_val

    def print_stats(self, scores):
        """
        Print macro/micro scores to stdout.
        """
        for task in self.tasks:
            print(f"{task:>12s}: {scores[task]['micro']:>10.4f}, {scores[task]['macro']:>10.4f}")
