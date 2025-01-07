"""
    Module predicting or scoring a trained model.
"""

# import sys
import datetime
import os
import pickle

import numpy as np
import pandas as pd
import scipy.special
import torch
import torchmetrics


class ScoreModel:
    """Class to score and make predictions from a trained model.

    Params:
        model_args: dict with model_keyword args, checked by caller.
        data_loaders: dict of PathReports class with split( train, test, val) as keys to
            score and predict on
        dw: DataHandler class, initialized in caller
        model: torch model fo predictiong and scoring
        device: torch device, either cpu or gpu, set in caller

    """

    def __init__(
        self, model_args, data_loaders, dw, model, device, savepath=None, clc_flag=False
    ):
        self.tasks = dw.num_classes.keys()
        self.abstain = model_args["abstain_kwargs"]["abstain_flag"]
        self.ntask = model_args["abstain_kwargs"]["ntask_flag"]
        self.mixed_precision = model_args["train_kwargs"]["mixed_precision"]
        self.clc = clc_flag

        self.model = model
        self.device = device
        self.data_loaders = data_loaders

        # setup loss function
        if self.abstain:
            reduction = "none"
        else:
            reduction = "mean"

        if self.clc:
            self.loss_fun = torch.nn.CrossEntropyLoss(reduction=reduction)
        else:
            self.multilabel = False  #  model_args["train_kwargs"]["multilabel"]
            if model_args["train_kwargs"]["class_weights"] is not None:
                with open(model_args["train_kwargs"]["class_weights"], "rb") as f:
                    weights = pickle.load(f)
                self.loss_funs = {}
                for task in self.tasks:
                    weights_task = torch.FloatTensor(weights[task]).to(self.device)
                    if self.multilabel:
                        self.loss_funs[task] = torch.nn.BCEWithLogitsLoss(
                            weights_task, reduction=reduction
                        )
                    else:
                        self.loss_funs[task] = torch.nn.CrossEntropyLoss(
                            weights_task, reduction=reduction
                        )
            else:
                self.loss_funs = {}
                for task in self.tasks:
                    if self.multilabel:
                        self.loss_funs[task] = torch.nn.BCEWithLogitsLoss(
                            None, reduction=reduction
                        )
                    else:
                        self.loss_funs[task] = torch.nn.CrossEntropyLoss(
                            None, reduction=reduction
                        )

        self.y_preds = {task: [] for task in self.tasks}
        self.y_trues = {task: [] for task in self.tasks if task != "Ntask"}
        self.logits = {task: [] for task in self.tasks}  # model output

        if savepath is None:
            self.savepath = "predictions/" + model_args["save_name"]
        else:
            tmp_path = savepath + "predictions/"
            if not os.path.exists(os.path.dirname(tmp_path)):
                os.makedirs(os.path.dirname(tmp_path))
            self.savepath = savepath + "predictions/" + model_args["save_name"]

        # define dicts for all scores
        self.scores = {}
        self.metrics = {}

    def score(self, data_loaders=None, split=None, dac=None, training_phase=False):
        """Score a dataset in a PathReport class.

        data_loaders: dict of PathReport class which provides X, y, and index of he entry in the
            original dataframe, to tie back to the metadata.
            The keys are 'train', 'test', and 'val' (if provided)
        split: string, one of 'train', 'test', or 'val' if wanting to score only a
            specific split
        training_phase - bool: sets model to train or eval mode
        dac - Abstention class: deep abstaining classifier class

        """
        if data_loaders is not None:
            print(
                (
                    "Error: score function requires a dict with keys 'train', 'test', or 'split'"
                    " and values PathReport classes."
                )
            )
            return None
        if split is not None:
            if isinstance(split, str) and split not in ["train", "test", "val"]:
                print(
                    (
                        "Error: score function requires a dict with keys 'train', 'test', or 'split'"
                        " and values PathReport classes."
                    )
                )
                return None

        if data_loaders is not None:
            data = data_loaders
        elif split is not None:
            data = self.data_loaders[split]
        elif data_loaders is None:
            data = self.data_loaders

        if split is None:
            for _split in data.keys():
                print(f"\nScoring {_split} set", flush=True)
                self.clear_pred_lists()
                self._score(
                    data[_split], _split, training_phase=training_phase, dac=dac
                )

                self.metrics[f"{_split}"] = {}
                self.metrics[f"{_split}"] = self.compute_scores(dac)
        else:
            print(f"\nScoring {split} set", flush=True)
            self._score(data, split, training_phase=training_phase, dac=dac)

            self.metrics[f"{split}"] = {}
            self.metrics[f"{split}"] = self.compute_scores(dac)

        self.save_scores()
        return None

    def _score(self, data_loader, split, training_phase=False, dac=None):
        """Score data_loader for validation.

        Args:
        data_loader - PathReports class
        split: str of data split, eg, train, test, val
        training_phase - bool: sets model to train or eval mode
        dac - Abstention class: deep abstaining classifier class

        Post condition:
            self.val_preds and self.val_trues updated.
            model set to train or eval depending on training_phase variable

        """
        losses = []

        if self.ntask:
            dac.ntask_filter = []

        if training_phase:
            self.model.train()
        else:
            self.model.eval()

        with torch.no_grad():
            for batch in data_loader:
                if self.clc:
                    X = batch["X"].to(self.device, non_blocking=True)
                    y = {
                        task: batch[f"y_{task}"].to(self.device)
                        for task in self.tasks
                        if task != "Ntask"
                    }
                    # max_seq_len = batch['X'].shape[1]
                    batch_len = batch["len"].to(self.device)

                    logits = self.model(X, batch_len)

                    mask = (
                        torch.arange(end=batch["X"].shape[1], device=self.device)[
                            None, :
                        ]
                        < batch_len[:, None]
                    )

                    idxs = torch.nonzero(mask, as_tuple=True)
                    loss = self.compute_clc_loss(logits, y, idxs, dac)
                    self.get_ys(logits, y, idxs)
                else:
                    y = {
                        task: batch[f"y_{task}"].cpu().numpy()
                        for task in self.tasks
                        if task != "Ntask"
                    }
                    loss, logits = self.compute_loss(batch, dac)
                    self.get_ys(logits, y)

                losses.append(loss.item())

        self.scores[f"{split}_loss"] = np.mean(losses)

    def compute_scores(self, dac=None):
        """Compute metrics to evaluate a model."""

        metrics = {}
        if self.abstain:
            pred_idxs = dac.compute_accuracy(self.y_trues, self.y_preds)

        # per task metrics
        for task in self.tasks:
            if task == "Ntask":
                continue
            logits = torch.vstack(self.logits[task]).to(self.device, non_blocking=True)
            if self.abstain:
                _trues_cpu = torch.tensor(self.y_trues[task], dtype=torch.int)[
                    pred_idxs[task]
                ]
                _preds_cpu = torch.tensor(self.y_preds[task], dtype=torch.int)[
                    pred_idxs[task]
                ]
                _trues = _trues_cpu.clone().to(self.device)
                _preds = _preds_cpu.clone().to(self.device)
            else:
                _trues = torch.tensor(self.y_trues[task], dtype=torch.int).to(
                    self.device
                )
                _preds = torch.tensor(self.y_preds[task], dtype=torch.int).to(
                    self.device
                )

            num_classes = logits.shape[1]
            if self.abstain:
                num_classes = num_classes - 1  # need to remove abstention class

            f1_micro = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="micro"
            ).to(self.device)
            metrics[f"{task}_micro"] = f1_micro(_preds, _trues)

            f1_macro = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(self.device)
            metrics[f"{task}_macro"] = f1_macro(_preds, _trues)

            accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(self.device)
            metrics[f"{task}_accuracy"] = accuracy(_preds, _trues)

            precision_micro = torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average="micro"
            ).to(self.device)
            metrics[f"{task}_precision_micro"] = precision_micro(_preds, _trues)

            precision_macro = torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(self.device)
            metrics[f"{task}_precision_macro"] = precision_macro(_preds, _trues)

            recall_micro = torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average="micro"
            ).to(self.device)
            metrics[f"{task}_recall_micro"] = recall_micro(_preds, _trues)

            recall_macro = torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(self.device)
            metrics[f"{task}_recall_macro"] = recall_macro(_preds, _trues)

        if self.abstain:
            _ = dac.compute_accuracy(self.y_trues, self.y_preds)
            for task in self.tasks:
                if task == "Ntask":
                    continue
                metrics[f"{task}_abs_acc"] = dac.accuracy[task]
            for k, v in dac.abs_rates.items():
                metrics[k] = v
            if self.ntask:
                ntask_scores = dac.compute_ntask_accuracy(self.y_trues, self.y_preds)
                metrics["ntask_acc"] = ntask_scores[0]
                metrics["ntask_abs_rate"] = ntask_scores[1]

        return metrics

    def save_scores(self):
        """Write scores (metrics), per task, to disk."""
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        with open(
            self.savepath + f"_metrics_{now}.csv", "w", encoding="utf-8"
        ) as f_out:
            for split, loss in self.scores.items():
                f_out.write(f"{split:10s}: {loss:7.6f}\n")
            f_out.write("\n")
            for split, split_dict in self.metrics.items():
                f_out.write(f"{split}\n")
                keys = sorted(split_dict.keys())
                for k, key in enumerate(keys):
                    f_out.write(f"{key:26s}: {split_dict[key]:7.6f} \n")
                    if k == len(keys):
                        f_out.write("\n")
                f_out.write("\n")

    def predict(
        self,
        metadata: dict,
        id2label: dict,
        data_loaders=None,
        save_probs=False,
        training_phase=False,
    ):
        """Make predictions from a trained model and save to disk.

        Params:
            metadata: dict of 'metadata' from the saved data frame to match predictions up
                with recordDocIds, etc.
            id2label: dict mapping between integers and labels, ie, C50
            data_loaders: PathReports class, with inputs, outputs, and indices in
                original dataFrame
            save_probs: bool, save the final softmax layer to disk
            training_phase: bool sets modelto train or eval
            dac - Abstention class: deep abstaining classifier class

        Postcondition:
            model set to eval or train

        If data_loaders is None, we will score each split, train, test, and val, otherwise it
        will just score one data_loader split.
        """

        if not isinstance(data_loaders, dict) and data_loaders is not None:
            print(
                (
                    "Error: predict function requires a dict with keys 'train', 'test', or 'split'"
                    " and values PathReport classes."
                )
            )
            return None
        if data_loaders is None:
            data = self.data_loaders
        else:
            data = data_loaders

        preds_df = []
        if save_probs:
            # df to which prob for each task will be appended
            probs_dfs = []

        for split in data.keys():
            print(f"\nPredicting {split} set", flush=True)
            preds = self._predict(
                data[split], save_probs, training_phase=training_phase
            )
            preds["split"] = split
            preds_df.append(self.preds_to_dataframe(preds, metadata[split], id2label))

            if save_probs:
                for task in self.tasks:
                    if task == "Ntask":
                        continue
                    preds[f"{task}_probs"] = preds["probs"][task]
                probs_dfs.append(
                    self.probs_to_dataframe(preds, metadata[split], id2label)
                )
        print("Saving predictions to a parquet file", flush=True)
        df = pd.concat(preds_df, axis=0)
        # if you want the index, set index=True
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        df.to_parquet(self.savepath + f"_preds_{now}.parquet",
                      engine='pyarrow',
                      compression='snappy',
                      index=False)

        if save_probs:
            print("Saving softmax scores to a parquet file")
            probs_final = pd.concat(probs_dfs, axis=0)
            probs_final.to_parquet(self.savepath + f"_probs_{now}.parquet",
                                   engine='pyarrow',
                                   compression='snappy',
                                   index=False)

        return None

    def _predict(self, data_loader, save_probs=False, training_phase=False):
        """Make predictions on data_loader.

        Params:
            data_laders: PathReports class, with inputs, outputs, and indices in
                original dataFrame
            save_probs: bool, save the final softmax layer to disk
            training_phase: bool sets modelto train or eval

        Postcondition:
            model set to eval or train

        """
        if training_phase:
            self.model.train()
        else:
            self.model.eval()

        preds = {}
        preds["pred_ys"] = {task: [] for task in self.tasks}
        preds["true_ys"] = {task: [] for task in self.tasks if task != "Ntask"}
        preds["idxs"] = []

        if save_probs:
            preds["probs"] = {task: [] for task in self.tasks if task != "Ntask"}
        if self.ntask:
            preds["ntask_probs"] = []

        with torch.no_grad():
            for batch in data_loader:
                if self.clc:
                    X = batch["X"].to(self.device)
                    y = {
                        task: batch[f"y_{task}"].to(self.device)
                        for task in self.tasks
                        if task != "Ntask"
                    }
                    batch_len = batch["len"].to(self.device)
                    max_seq_len = X.shape[1]
                    logits = self.model(X, batch_len)

                    mask = (
                        torch.arange(end=max_seq_len, device=self.device)[None, :]
                        < batch_len[:, None]
                    )
                    idxs = torch.nonzero(mask, as_tuple=True)
                    _idxs = batch["index"][mask.cpu()]
                    data_idxs = [i.item() for i in _idxs]

                    # list of indices in orginal DataFrame
                    preds["idxs"].extend(data_idxs)
                    preds = self.get_clc_preds(logits, y, idxs, preds, save_probs)
                else:
                    preds["idxs"].extend(batch["index"].tolist())
                    preds = self.get_preds(batch, preds, save_probs)

        return preds

    def preds_to_dataframe(self, preds: dict, metadata: pd.DataFrame, id2label: dict):
        """Save test set predictions as DataFrame.

        Params:
            probs: dict with softmax scores as values and split as keys
            metadata: DataFrame with recordDocId,... from generated data
            id2label: dict with int -> str mappings, ie, 43 -> C50
            save_probs: bool, save the final softmax layer to disk

        Returns: pd.DataFrame with metadata, predicted and true classes.
        """

        ntask_map = {0: "predict", 1: "abstain"}
        cols = {}
        df_list = []
        col_list = metadata.columns.tolist()

        for task in self.tasks:
            if task == "Ntask":
                cols["Ntask score"] = preds["pred_ys"]["Ntask"]
                col_ids = ["Ntask score", "Ntask pred"]
                df = pd.DataFrame(cols, columns=col_ids, index=preds["idxs"])
                df["Ntask pred"] = df["Ntask score"].apply(lambda x: ntask_map[round(x)])

            else:
                cols[f"{task}_true"] = preds["true_ys"][task]
                cols[f"{task}_pred"] = [i.tolist() for i in preds["pred_ys"][task]]
                col_ids = [f"{task}_true", f"{task}_pred"]

                df = pd.DataFrame(cols, columns=col_ids, index=preds["idxs"])

                df[f"{task}_true"] = df[f"{task}_true"].map(id2label[task])
                df[f"{task}_pred"] = df[f"{task}_pred"].map(id2label[task])

            df_list.append(df)
            col_list.extend(col_ids)
        all_preds = pd.concat(df_list, axis=1)
        final_df = all_preds.merge(metadata, left_index=True, right_index=True)
        final_df.sort_index(inplace=True)
        final_df = final_df[col_list]

        return final_df

    def probs_to_dataframe(self, preds: dict, metadata: pd.DataFrame, id2label: dict):
        """Create dataframe with softmax scores and metadata from generated data.

        Params:
            preds: dict with indices matching to original data and ground truth labels
            metadata: DataFrame with recordDocId,... from generated data
            id2label: dict with int -> str mappings, ie, 43 -> C50

        Returns: pd.DataFrame with metadata and softmax scores
        """

        cols = {}
        temp = {}
        df_list = []
        df = None
        col_list = metadata.columns.tolist()

        for task in self.tasks:
            if task == "Ntask":
                continue

            cols[f"{task}_true"] = preds["true_ys"][task]
            cols[f"{task}_pred"] = [i.tolist() for i in preds["pred_ys"][task]]
            temp[f"{task}_probs"] = [i.tolist() for i in preds["probs"][task]]
            max_classes = len(temp[f"{task}_probs"][0])

            # case where there are more than 5 possible classes and we return top 5
            if max_classes > 5:
                max_classes = 5

            top_preds_list = []
            top_probs_list = []
            for i in range(len(temp[f"{task}_probs"])):
                probs_arr = np.array(temp[f"{task}_probs"][i])
                # indexes with top values
                top_indexes = np.argsort(-probs_arr)[:max_classes]
                # top softmaxes, round to 6th decimals
                top_probs = probs_arr[top_indexes].round(6)
                # add for each instance that is predicted on
                top_preds_list.append(top_indexes)
                top_probs_list.append(top_probs)

            cols[f"{task}_top_preds"] = top_preds_list
            cols[f"{task}_top_probs"] = top_probs_list

            col_ids = [f"{task}_true", f"{task}_pred", f"{task}_top_preds", f"{task}_top_probs" ]

            df = pd.DataFrame(cols, columns=col_ids, index=preds["idxs"])
            df[f"{task}_true"] = df[f"{task}_true"].map(id2label[task])
            df[f"{task}_pred"] = df[f"{task}_pred"].map(id2label[task])

            if isinstance(df[f"{task}_top_preds"].iloc[0], np.int64):
                df[f"{task}_top_preds"] = df[f"{task}_top_preds"].map(id2label[task])
            else:
                df[f"{task}_top_preds"] = df[f"{task}_top_preds"].apply(lambda idx: [id2label[task][i] for i in idx])

            df_list.append(df)
            col_list.extend(col_ids)

        all_probs = pd.concat(df_list, axis=1)
        final_df = all_probs.merge(metadata, left_index=True, right_index=True)
        final_df.sort_index(inplace=True)
        final_df = final_df[col_list]

        return final_df

    def evaluate_model(
        self,
        metadata: dict,
        id2label: dict,
        data_loaders=None,
        save_probs=False,
        dac=None,
        training_phase=False,
    ):
        """Score and predict from trained model

        Params:
             metadata: dict of 'metadata' from the saved data frame to match predictions up
                 with recordDocIds, etc.
             id2label: dict mapping between integers and labels, ie, C50
             data_laders: PathReports class, with inputs, outputs, and indices in
                 original dataFrame
             save_probs: bool, save the final softmax layer to disk
             dac - Abstention class: deep abstaining classifier class
             training_phase: bool sets modelto train or eval

         Postcondition:
             model set to eval or train

         If data_loaders is None, we will score eahc split, train, test, and val, otherwise it
         will just score one data_loader split.

        """
        if not isinstance(data_loaders, dict) and data_loaders is not None:
            print(
                (
                    "Error: predict function requires a dict with keys 'train', 'test', or 'split'"
                    " and values PathReport classes."
                )
            )
            return
        if data_loaders is None:
            data = self.data_loaders
        else:
            data = data_loaders

        preds_df = []

        if save_probs:
            probs_df = []

        for split in data.keys():
            print(f"\nEvaluating {split} set", flush=True)
            self.clear_pred_lists()
            preds = self.pred_and_score(
                data[split],
                split,
                save_probs=save_probs,
                dac=dac,
                training_phase=training_phase,
            )
            preds["split"] = split
            preds_df.append(self.preds_to_dataframe(preds, metadata[split], id2label))
            if save_probs:
                for task in self.tasks:
                    if task == "Ntask":
                        continue
                    preds[f"{task}_probs"] = preds["probs"][task]
                probs_df.append(
                    self.probs_to_dataframe(preds, metadata[split], id2label)
                )

            self.metrics[f"{split}"] = {}
            self.metrics[f"{split}"] = self.compute_scores(dac)

        self.save_scores()

        print("Saving predictions to csv", flush=True)
        all_preds = pd.concat(preds_df, axis=0)
        all_preds.sort_index(inplace=True)

        # if you want the index, set index=True
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        all_preds.to_csv(
            self.savepath + f"_preds_{now}.csv", float_format="%7.6f", index=False
        )

        if save_probs:
            print("Saving softmax scores to csv")
            all_probs = pd.concat(probs_df, axis=0)
            all_probs.sort_index(inplace=True)

            # if you want the index, set index=True
            all_probs.to_csv(
                self.savepath + f"_probs_{now}.csv", float_format="%7.6f", index=False
            )

    def pred_and_score(
        self, data_loader, split, save_probs=False, dac=None, training_phase=False
    ):
        """Make predictions and score from a trained model.

        Params:
            data_laders: PathReports class, with inputs, outputs, and indices in
                original dataFrame
            split: string representation of twhich split to pred and score, ie train,
                test, or val
            save_probs: bool, save the final softmax layer to disk
            dac - Abstention class: deep abstaining classifier class
            training_phase: bool sets modelto train or eval

        Postcondition:
            model set to eval or train
            self.logits, self.y_preds, and self.y_trues populated

        """
        if training_phase:
            self.model.train()
        else:
            self.model.eval()

        losses = []
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        preds = {}
        preds[
            "pred_ys"
        ] = self.y_preds  # {task: [] for task in self.tasks if task != "Ntask"}
        preds[
            "true_ys"
        ] = self.y_trues  # {task: [] for task in self.tasks if task != "Ntask"}
        preds["idxs"] = []

        if save_probs:
            preds["probs"] = {task: [] for task in self.tasks if task != "Ntask"}

        if self.ntask:
            preds["ntask_probs"] = []
            dac.ntask_filter = []

        with torch.no_grad():
            for batch in data_loader:
                if self.clc:
                    X = batch["X"].to(self.device)
                    y = {
                        task: batch[f"y_{task}"].to(self.device)
                        for task in self.tasks
                        if task != "Ntask"
                    }
                    batch_len = batch["len"].to(self.device)
                    max_seq_len = X.shape[1]
                    logits = self.model(X, batch_len)

                    mask = (
                        torch.arange(end=max_seq_len, device=self.device)[None, :]
                        < batch_len[:, None]
                    )
                    idxs = torch.nonzero(mask, as_tuple=True)
                    _idxs = batch["index"][mask.cpu()]
                    data_idxs = [i.item() for i in _idxs]
                    # list of indices in orginal DataFrame
                    preds["idxs"].extend(data_idxs)

                    preds = self.get_clc_preds(logits, y, idxs, preds, save_probs)
                    loss = self.compute_clc_loss(logits, y, idxs, dac)
                    for task in self.tasks:
                        self.logits[task].extend(logits[task].cpu())
                else:
                    preds["idxs"].extend(batch["index"].tolist())
                    preds = self.get_preds(batch, preds, save_probs)
                    loss, logits = self.compute_loss(batch, dac)
                    for task in self.tasks:
                        self.logits[task].extend(logits[task])

                losses.append(loss.item())

        self.scores[f"{split}_loss"] = np.mean(losses)

        return preds

    def get_ys(self, logits, y, idxs=None):
        """Get ground truth and y_prediction lists.

        Args:
            logits: torch device tensor from model.forward
            y: torch device tensor of ground truth values
            idxs: torch tensor on indices for clc predictions, optional

            Post-condition:
                self.y_trues and self.y_preds populated.

        """
        for task in self.tasks:
            if task == "Ntask":
                self.y_preds[task].extend(torch.sigmoid(logits[task]).flatten().cpu().tolist())
                continue
            if self.clc:
                self.logits[task].extend(logits[task][idxs])
                self.y_preds[task].extend(torch.argmax(logits[task][idxs], axis=1))
                self.y_trues[task].extend(y[task][idxs].detach().cpu().tolist())
            else:
                self.logits[task].extend(logits[task])
                self.y_preds[task].extend(torch.argmax(logits[task], axis=1))
                if self.multilabel:
                    self.y_trues[task].extend(np.argmax(y[task]), axis=1)
                else:
                    self.y_trues[task].extend(y[task])

    def clear_pred_lists(self):
        """Delete entries in the lists to avoid re-initialization.

        Post-condition:
            Entries in self.y_trues and self.y_preds deleted.

        """
        for task in self.tasks:
            if task != "Ntask":
                del self.y_trues[task][:]
            del self.logits[task][:]
            del self.y_preds[task][:]

    def compute_loss(self, batch, dac=None):
        """Compute forward pass and loss function.

        Args:, requires_grad=True
            batch - torch iterate from DataLoader
            dac: deep abstaining classifier class
            ntask_abs: float, probability of abstaining on entire document

        Returns:
            loss - torch.tensor(float)

        Post-condition:
            y_preds and y_trues populated
            loss updated
        """
        X = batch["X"].to(self.device)
        y = {
            task: batch[f"y_{task}"].to(self.device)
            for task in self.tasks
            if task != "Ntask"
        }
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            logits = self.model(X)

            loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if self.ntask:
                ntask_abs = torch.sigmoid(logits["Ntask"])[:, -1]
                dac.get_ntask_filter(ntask_abs)

            for task in self.tasks:
                if task == "Ntask":
                    continue
                if self.ntask:
                    if task in dac.ntask_tasks:
                        loss += dac.abstention_loss(
                            logits[task], y[task], task, ntask_abs_prob=ntask_abs
                        )
                    else:
                        loss += dac.abstention_loss(logits[task], y[task], task)
                elif self.abstain:  # just the dac
                    loss += dac.abstention_loss(logits[task], y[task], task)
                else:  # nothing fancy
                    loss += self.loss_funs[task](logits[task], y[task])

            if self.ntask:
                loss = loss - torch.mean(
                    dac.ntask_alpha * torch.log(1 - ntask_abs + 1e-6)
                )
        # average over all tasks
        return loss / len(self.tasks), logits

    def compute_clc_loss(self, logits, y, idxs, dac=None):
        """Compute forward pass and case level loss function.

        Args:, requires_grad=True)
            batch - torch iterate from DataLoader
            dac: deep abstaining classifier class
            ntask_abs: float, probability of abstaining on entire document

        Returns:
            loss - torch.tensor(float)

        Post-condition:
            y_preds and y_trues populated
            loss updated
        """

        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if self.ntask:
            ntask_abs = torch.sigmoid(logits["Ntask"][idxs])[:, -1]
            dac.get_ntask_filter(ntask_abs)

        for task in self.tasks:
            if task == "Ntask":
                continue
            y_true = y[task][idxs]
            y_pred = logits[task][idxs]

            if self.ntask:
                if task in dac.ntask_tasks:
                    loss += dac.abstention_loss(
                        y_pred, y_true, task, ntask_abs_prob=ntask_abs
                    )
                else:
                    loss += dac.abstention_loss(y_pred, y_true, task)
            elif self.abstain:  # just the dac
                loss += dac.abstention_loss(y_pred, y_true, task)
            else:  # nothing fancy
                loss += self.loss_fun(y_pred, y_true)

        if self.ntask:
            loss = loss - torch.mean(dac.ntask_alpha * torch.log(1 - ntask_abs + 1e-6))
        # average over all tasks
        return loss / len(self.tasks)

    def get_preds(self, batch, preds, save_probs=False):
        """Compute forward pass and predictions.

        Args:, requires_grad=True)
            batch - torch iterate from DataLoader
            preds - dict of lists of predictions and ground truth values
            save_probs: bool, save the final softmax layer to disk

        Returns:
           None

        Post-condition:
            y_preds and y_trues populated
        """
        X = batch["X"].to(self.device)
        logits = self.model(X)

        for task in self.tasks:
            if task == "Ntask":
                preds["pred_ys"]["Ntask"].extend(torch.sigmoid(logits[task])[:, -1].detach().cpu().tolist())
                continue
            outputs = logits[task].detach().cpu().numpy()
            preds["pred_ys"][task].extend(np.argmax(outputs, axis=1))
            if f"y_{task}" in batch.keys():
                preds["true_ys"][task].extend(batch[f"y_{task}"].tolist())

            if save_probs:
                if self.multilabel:
                    preds["probs"][task].extend(
                        scipy.special.expit(outputs)
                    )  # logistic sigmoid
                else:
                    preds["probs"][task].extend(scipy.special.softmax(outputs, axis=1))

        return preds

    def get_clc_preds(self, logits, y, idxs, preds, save_probs=False):
        """Compute forward pass and predict from case level context.

        Args:, requires_grad=True)
            batch - torch iterate from DataLoader

        Returns:
            None

        Post-condition:
            preds keys 'y_preds' and 'y_trues' populated
        """

        for task in self.tasks:
            if task == "Ntask":
                ntask_prob = torch.sigmoid(logits["Ntask"][idxs])[:, -1].detach().cpu().tolist()
                preds["pred_ys"]["Ntask"].extend(ntask_prob)
            else:
                _preds = logits[task][idxs].detach().cpu().numpy()
                preds["pred_ys"][task].extend(np.argmax(_preds, axis=1))
                preds["true_ys"][task].extend(y[task][idxs].tolist())
                if save_probs:
                    preds["probs"][task].extend(scipy.special.softmax(_preds, axis=1))

        return preds
