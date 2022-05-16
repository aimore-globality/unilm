from __future__ import absolute_import, division, print_function

from pprint import pprint
import logging
from pathlib import Path
from typing import List, Dict
import torch

from markuplmft.fine_tuning.run_swde.utils import set_seed
from markuplmft.fine_tuning.run_swde import constants
from markuplmft.models.markuplm import (
    MarkupLMConfig,
    MarkupLMForTokenClassification,
    MarkupLMTokenizer,
)

logger = logging.getLogger(__name__)


class MarkupLModel:
    def __init__(
        self,
        pre_trained_model_folder_path: str = "/data/GIT/unilm/markuplm/markuplmft/models/markuplm/286",
        verbose: bool = False,
        local_rank: int = -1,
        device=None,
        n_gpu: int = -1,
        label_smoothing: float = 0.0,
        loss_function="CrossEntropy",
    ):
        self.net = None
        self.tokenizer = None
        self.label_smoothing = label_smoothing
        self.local_rank = local_rank

        self.device = device
        self.n_gpu = n_gpu
        set_seed(self.n_gpu)  # ? For reproducibility

        self.pre_trained_model_folder_path = pre_trained_model_folder_path

        # self.output_dir = Path("/data/GIT/unilm/markuplm/markuplmft/models/markuplm/")

        self.original_model_dir = Path("microsoft/markuplm-base")
        self.save_model_path = Path("/data/GIT/unilm/markuplm/markuplmft/models/markuplm/")

        self.doc_stride = 256 # 128
        self.max_seq_length = 384

        # TODO (AIMORE): Change back to default values
        print(f"doc_stride: {self.doc_stride}" )
        print(f"max_seq_length: {self.max_seq_length}" )

        self.loss_function = loss_function

        if verbose:
            print("self.__dict__", pprint(self.__dict__))

    def load_pretrained_model_and_tokenizer(self):
        # ? Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()

            # ? Make sure only the first process in distributed training will download model & vocab
        self.tokenizer = MarkupLMTokenizer.from_pretrained(self.original_model_dir)

        config = MarkupLMConfig.from_pretrained(self.original_model_dir)
        config_dict = config.to_dict()
        config_dict.update({"node_type_size": len(constants.ATTRIBUTES_PLUS_NONE)})
        config = MarkupLMConfig.from_dict(config_dict)

        self.net = MarkupLMForTokenClassification.from_pretrained(
            self.original_model_dir,
            label_smoothing=self.label_smoothing,
            loss_function=self.loss_function,
            config=config,
        )
        self.net.resize_token_embeddings(len(self.tokenizer))

    def load_trained_model(
        self,
        config_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
        tokenizer_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
        net_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/1/checkpoint-1",
    ):

        config = MarkupLMConfig.from_pretrained(config_path)
        self.tokenizer = MarkupLMTokenizer.from_pretrained(tokenizer_path)
        self.net = MarkupLMForTokenClassification.from_pretrained(
            net_path,
            label_smoothing=self.label_smoothing,
            loss_function=self.loss_function,
            config=config,
        )
        self.net.to(self.device)

    def save_model_and_tokenizer(self):
        # TODO (aimore): Replace os with Path [done - remove comment if passes]
        # if not os.path.exists(self.save_model_path) and self.local_rank in [-1, 0]:
        #     os.makedirs(self.save_model_path)
        if self.local_rank in [-1, 0]:
            self.save_model_path.mkdir(parents=True, exist_ok=False)  #! Uncomment

        print(f"Saving model checkpoint to {self.save_model_path}")
        # ? Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # ? They can then be reloaded using `from_pretrained()`
        # ? Take care of distributed/parallel training
        model_to_save = self.net.module if hasattr(self.net, "module") else self.net
        model_to_save.save_pretrained(self.save_model_path)
        self.tokenizer.save_pretrained(self.save_model_path)
        # TODO(aimore): Save the parameters: torch.save(self.__dict__, os.path.join(self.save_model_path, "training_args.bin"))

    def save_model(self, save_dir, epoch):
        self.save_path = Path(save_dir) / f"checkpoint-{epoch}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        model_to_save = self.net.module if hasattr(self.net, "module") else self.net
        # ? Take care of distributed/parallel training
        model_to_save.save_pretrained(self.save_path)
        #! Save the parameters: torch.save(self.__dict__, os.path.join(output_dir, "training_args.bin"))
        if self.save_path.exists():
            print(f"Overwriting model checkpoint: {self.save_path}")
        else:
            print(f"Saving model checkpoint to: {self.save_path}")

    def zero_grad(self):
        """
        According to: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#:~:text=for%20param%20in%20model.parameters()%3A%0A%20%20%20%20param.grad%20%3D%20None
        this method of zeroing the gradients is preferable, because it reduces the number of memory operations.
        """
        for param in self.net.parameters():
            param.grad = None

    def freeze_body(self, verbose=False):
        print("Freezing Model's Body")
        for name, module in self.net.named_modules():
            if name in ["token_cls", "token_cls.dense", "token_cls.LayerNorm", "token_cls.decoder"]:
                if name == "token_cls":
                    for param in module.parameters():
                        param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = True
            if verbose:
                print(f"{name}:  {module.parameters}")
                print()
