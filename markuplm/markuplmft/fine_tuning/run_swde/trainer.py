from typing import Dict, List, Sequence
from grpc import access_token_call_credentials
from markuplmft.fine_tuning.run_swde.featurizer import Featurizer
from markuplmft.fine_tuning.run_swde.classifier import NodeClassifier
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset
from torch.utils.data import DataLoader
import glob
from transformers import set_seed
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from transformers import AdamW
import transformers

set_seed(66)


class Trainer:
    def __init__(
        self,
        featurizer,
        classifier,
        train_df,
        train_batch_size: int,
        evaluate_df,
        evaluate_batch_size: int,
        num_epochs: int,
        save_model_dir: str,
        overwrite_model: bool = False,
        evaluate_during_training: bool = False,
        weight_decay: float = 0.01,
        learning_rate: float = 1e-05,
        adam_epsilon: float = 1e-8,
        warmup_ratio: float = 0,
    ):
        self.featurizer: Featurizer = featurizer
        self.classifier: NodeClassifier = classifier

        # ? Setting Data
        self.train_df = train_df.copy()
        self.evaluate_df = evaluate_df.copy()

        self.train_batch_size = train_batch_size
        self.evaluate_batch_size = evaluate_batch_size

        # ? Setting Training
        self.num_epochs = num_epochs
        self.evaluate_during_training = evaluate_during_training
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon

        # ? Setting Model
        self.overwrite_model = overwrite_model
        self.save_model_dir = save_model_dir
        self.training_samples = len(train_df)
        self.evaluation_samples = len(evaluate_df)

    def prepare_data(self, accelerator, df, dataset_name="develop"):
        accelerator.print(f"Converting features to dataset {dataset_name}...")
        dataset_features = self.featurizer.feature_to_dataset(df["page_features"].explode().values)
        shuffle = False

        if dataset_name == "train":
            batch_size = self.train_batch_size
            self.train_features = dataset_features
            shuffle = True    
        else:
            batch_size = self.evaluate_batch_size
            self.evaluate_features = dataset_features

        dataloader = DataLoader(
            dataset=dataset_features,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4,
        )

        accelerator.print(
            f"Samples: {len(dataset_features)} | Batches: {len(dataloader)} | Batch_size = {batch_size}"
        )
        return dataloader

    def prepare_for_training(self, accelerator):
        self.train_dataloader = self.prepare_data(accelerator, self.train_df, dataset_name="train")
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        accelerator.print(f"Num Epochs = {self.num_epochs}")
        accelerator.print(f"Num Training steps = {self.num_training_steps}")
        accelerator.print(f"Num training data points = {len(self.train_df)}")

        self.evaluate_dataloader = self.prepare_data(accelerator, self.evaluate_df)

        # accelerator.log.watch(self.classifier.model)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for param_name, param in self.classifier.model.named_parameters()
                    if not any(nd in param_name for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param
                    for param_name, param in self.classifier.model.named_parameters()
                    if any(nd in param_name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        (
            self.classifier.model,
            self.optimizer,
            self.train_dataloader,
            self.evaluate_dataloader,
        ) = accelerator.prepare(
            self.classifier.model,
            self.optimizer,
            self.train_dataloader,
            self.evaluate_dataloader,
        )

        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=100,
            num_training_steps=self.num_training_steps,
        )

    def train_epoch(self, accelerator):
        all_losses = []
        self.classifier.model.train()
        batch_progress_bar = tqdm(
            self.train_dataloader, disable=not accelerator.is_main_process, desc="Batch"
        )
        for train_step, batch in enumerate(batch_progress_bar):
            outputs = self.classifier.model(**batch)
            loss = outputs[0]
            accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            all_losses.append(loss.item())
        return all_losses

    def train(self, accelerator):
        accelerator.print(f"training_pages: {self.training_samples}")
        accelerator.print(f"evaluation_pages: {self.evaluation_samples}")

        if accelerator.is_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()

        self.prepare_for_training(accelerator)

        #! Training step
        accelerator.print("Train...")
        epoch_progress_bar = tqdm(
            range(self.num_epochs), disable=not accelerator.is_main_process, desc="Epoch"
        )
        for epoch in epoch_progress_bar:
            if self.evaluate_during_training:
                self.evaluate(accelerator=accelerator)
            all_losses = self.train_epoch(accelerator=accelerator)
            loss_avg = np.mean(all_losses)
            accelerator.print(f"Epoch: {epoch} - Loss: {loss_avg}")
            accelerator.log({"train/loss": loss_avg, "this": 1})

            epoch_progress_bar.update(1)

        self.evaluate(accelerator=accelerator)

        #! Save model
        if self.overwrite_model:
            accelerator.print(f"Overwritting model...")
            save_model_path = self.save_model_dir + "pytorch_model_1.pth"
            save_pred_data_path = self.save_model_dir + "develop_pred.pkl"
            accelerator.save_state(self.save_model_dir)
            accelerator.print(f"Saved model at:\n {save_model_path}")
            accelerator.print(f"Saved predicted data at:\n {save_pred_data_path}")
            self.evaluate_df.drop(["page_features"], axis=1).to_pickle(save_pred_data_path)

    def evaluate(self, accelerator):
        accelerator.print("Evaluate...")
        self.classifier.model.eval()

        batch_progress_bar = tqdm(
            self.evaluate_dataloader, disable=not accelerator.is_main_process, desc="Batch"
        )
        all_logits = []
        all_losses = []
        for batch in batch_progress_bar:
            with torch.no_grad():
                outputs = self.classifier.model(**batch)
            loss = outputs[0]
            logits = outputs.logits
            all_losses.append(loss.item())
            all_logits.append(accelerator.gather(logits).detach().cpu())

        loss_avg = np.mean(all_losses)
        accelerator.print(f"Evaluation - Loss: {loss_avg}")
        accelerator.log({"evaluation/loss": loss_avg, "this": 1})

        urls = self.evaluate_features.urls
        node_ids = self.evaluate_features.node_ids
        all_probs = torch.softmax(torch.cat(all_logits, dim=0), dim=2)[:, :, 0]
        selected_probs = [np.array(all_probs[e, : len(node_id)]) for e, node_id in enumerate(node_ids)]
        urls_node_ids_selected_probs = pd.DataFrame(
            zip(urls, node_ids, selected_probs), columns=["urls", "node_ids", "selected_probs"]
        )
        sorted_selected_probs = (
            urls_node_ids_selected_probs.explode(["node_ids", "selected_probs"])
            .groupby(["urls", "node_ids"])
            .agg(list)
            .sort_values(["urls", "node_ids"])
        )
        if trainer_config['method'] == 'mean':
            node_probs = sorted_selected_probs.apply(
                lambda probs: np.mean(probs["selected_probs"]), axis=1
            ).values
        else:
            node_probs = sorted_selected_probs.apply(
                lambda x: np.max(x["selected_probs"]), axis=1
            ).values

        # TODO: Add some function to encapsulate these lines
        self.evaluate_df["html"] = self.evaluate_df["html"].astype("category")
        # accelerator.print(f"Memory: {sum(self.evaluate_df.memory_usage(deep=True))/10**6:.2f} Mb")
        evaluate_nodes_df = (
            self.evaluate_df.explode("nodes", ignore_index=True).reset_index().copy()
        )
        evaluate_nodes_df = evaluate_nodes_df.join(
            pd.DataFrame(
                evaluate_nodes_df.pop("nodes").tolist(),
                columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
            )
        )
        evaluate_nodes_df = evaluate_nodes_df.sort_values(["url", "index"])
        assert np.all(
            [x[0] for x in sorted_selected_probs.index] == evaluate_nodes_df["url"].values
        )
        assert len(node_probs) == len(evaluate_nodes_df)

        # accelerator.print(f"Memory: {sum(evaluate_df.memory_usage(deep=True))/10**6:.2f} Mb")
        evaluate_nodes_df["node_prob"] = node_probs
        evaluate_nodes_df["node_pred"] = node_probs > self.classifier.decision_threshold

        # TODO: move this out
        evaluate_nodes_df["node_gt"] = evaluate_nodes_df["node_gt_tag"] == "PAST_CLIENT"
        evaluate_nodes_df["node_pred_tag"] = evaluate_nodes_df["node_pred"].apply(
            lambda node_pred: "PAST_CLIENT" if node_pred else "none"
        )

        accelerator.print("Compute Metrics:")
        metrics_per_dataset, cm_per_dataset = compute_metrics_per_dataset(evaluate_nodes_df)
        accelerator.print(
            f"Node Classification Metrics per Dataset:\n {metrics_per_dataset} | cm_per_dataset: {cm_per_dataset}"
        )
        precision, recall, f1 = (
            metrics_per_dataset["precision"],
            metrics_per_dataset["recall"],
            metrics_per_dataset["f1"],
        )
        accelerator.log(
            {
                "evaluation/precision": precision,
                "evaluation/recall": recall,
                "evaluation/f1": f1,
                "this": 1,
            }
        )
        accelerator.print("... Evaluation Done")
        return evaluate_nodes_df

    def infer(self, accelerator):
        #! Load model
        accelerator.print(f"Loading state from: {self.save_model_dir}")
        self.evaluate_dataloader = self.prepare_data(accelerator, self.evaluate_df)

        (
            self.classifier.model,
            self.evaluate_dataloader,
        ) = accelerator.prepare(
            self.classifier.model,
            self.evaluate_dataloader,
        )

        accelerator.load_state(self.save_model_dir)

        device = accelerator.device
        accelerator.print(f"device: {device}")

        accelerator.print(f"Model State - loaded:")

        evaluate_df = self.evaluate(accelerator)
        accelerator.print("... Infer Done")

        #! Save infered dataset
        save_path = self.save_model_dir + "develop_df_pred.pkl"
        accelerator.print(f"Saved infered data at: {save_path}")
        evaluate_df.drop(["page_features"], axis=1).to_pickle(save_path)


if __name__ == "__main__":
    trainer_config = dict(
        name_root_folder="delete-abs",
        dataset_to_use="all",
        num_epochs=3,
        train_batch_size=28,
        evaluate_batch_size=28 * 10,
        save_model_dir="/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/",
        evaluate_during_training=True,
        overwrite_model=True,
        with_img=False,
        method="max",
    )

    if trainer_config["dataset_to_use"] == "mini":
        trainer_config["num_epochs"] = 4

    if trainer_config["dataset_to_use"] == "debug":
        trainer_config["num_epochs"] = 1
        trainer_config["train_batch_size"], trainer_config["evaluate_batch_size"] = 28, 28 * 10

    if trainer_config["with_img"]:
        trainer_config["name_root_folder"] = "delete-img"
        trainer_config["save_model_dir"] = trainer_config["save_model_dir"].replace(
            "models", "models_with_img"
        )

    train_data_path = (
        f"/data/GIT/{trainer_config['name_root_folder']}/train/processed_dedup_feat.pkl"
    )
    develop_data_path = (
        f"/data/GIT/{trainer_config['name_root_folder']}/develop/processed_dedup_feat.pkl"
    )

    df_train = pd.read_pickle(train_data_path)
    df_develop = pd.read_pickle(develop_data_path)

    # ?  Mini
    if trainer_config["dataset_to_use"] == "mini":
        df_train = df_train[df_train.domain.isin(df_train.domain.unique()[:24])]
        df_develop = df_develop[df_develop.domain.isin(df_develop.domain.unique()[:24])]

    # ?  Debug
    elif trainer_config["dataset_to_use"] == "debug":
        df_train = df_train[df_train.domain.isin(df_train.domain.unique()[:4])]
        df_develop = df_develop[df_develop.domain.isin(df_develop.domain.unique()[:4])]

    accelerator = Accelerator(log_with="wandb")
    accelerator.print(f"device: {accelerator.device}")

    if accelerator.state.process_index == 0:
        accelerator.init_trackers(project_name="LanguageModel", config=trainer_config)

    featurizer = Featurizer()

    classifier_config = dict(decision_threshold=0.5)
    classifier = NodeClassifier(**classifier_config)

    trainer = Trainer(
        featurizer=featurizer,
        classifier=classifier,
        save_model_dir=trainer_config["save_model_dir"],
        train_df=df_train,
        train_batch_size=trainer_config["train_batch_size"],
        evaluate_df=df_develop,
        evaluate_batch_size=trainer_config["evaluate_batch_size"],
        evaluate_during_training=trainer_config["evaluate_during_training"],
        num_epochs=trainer_config["num_epochs"],
        overwrite_model=trainer_config["overwrite_model"],
    )

    accelerator.print(f"train: {len(df_train)} - {train_data_path}")
    accelerator.print(f"develop: {len(df_develop)} - {develop_data_path}")

    trainer.train(accelerator)
    accelerator.end_training()
    # trainer.infer(accelerator)
