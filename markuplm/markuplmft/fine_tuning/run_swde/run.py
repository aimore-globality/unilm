from __future__ import absolute_import, division, print_function
import multiprocessing as mp
import argparse
import collections
# import copy
import glob
import logging
import os
import random
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import torch
# from markuplmft.fine_tuning.run_swde.eval_utils import page_level_constraint
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from markuplmft.fine_tuning.run_swde.utils import SwdeDataset, get_swde_features

from markuplmft.fine_tuning.run_swde import constants
from markuplmft.models.markuplm import (
    MarkupLMConfig,
    MarkupLMForTokenClassification,
    MarkupLMTokenizer,
)
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
    """
    Fix the random seed for reproduction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(
    local_rank,
    per_gpu_train_batch_size,
    n_gpu,
    max_steps,
    num_train_epochs,
    gradient_accumulation_steps,
    weight_decay,
    learning_rate,
    adam_epsilon,
    warmup_ratio,
    fp16,
    fp16_opt_level,
    seed,
    device,
    max_grad_norm,
    logging_steps,
    evaluate_during_training,
    save_steps,
    train_dataset, 
    model, 
    sub_output_dir, 
    **kwargs
    ):

    """
    Train the model
    """
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    else:
        tb_writer = None

    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
    )

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = (
            max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # TODO (Aimore): Improve the name of these variables
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                param for param_name, param in model.named_parameters() if not any(nd in param_name for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param for param_name, param in model.named_parameters() if any(nd in param_name for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_ratio * t_total), num_training_steps=t_total
    )
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {per_gpu_train_batch_size}")

    total_train_batch_size = train_batch_size * gradient_accumulation_steps * (torch.distributed.get_world_size() if local_rank != -1 else 1)
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0]
    )
    set_seed(seed, n_gpu)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in train_iterator:
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch_list in enumerate(epoch_iterator):
            model.train()
            batch_list = tuple(batch.to(device) for batch in batch_list)
            inputs = {
                "input_ids": batch_list[0],
                "attention_mask": batch_list[1],
                "token_type_ids": batch_list[2],
                "xpath_tags_seq": batch_list[3],
                "xpath_subs_seq": batch_list[4],
                "labels": batch_list[5],
            }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if n_gpu > 1:
                loss = loss.mean()  # to average on multi-gpu parallel (not distributed) training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16: # TODO (Aimore): Replace this with Accelerate
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    local_rank in [-1, 0]
                    and logging_steps > 0
                    and global_step % logging_steps == 0
                ):
                    # Log metrics
                    if local_rank == -1 and evaluate_during_training:
                        raise ValueError("Shouldn't `evaluate_during_training` when ft SWDE!!")
                        # results = evaluate(args, model, tokenizer, prefix=str(global_step))
                        # for key, value in results.items():
                        #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                if (
                    local_rank in [-1, 0]
                    and save_steps > 0
                    and global_step % save_steps == 0
                ):
                    # Save model checkpoint
                    save_model(sub_output_dir, model, global_step)

            if 0 < max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < max_steps < global_step:
            train_iterator.close()
            break

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def save_model(sub_output_dir, model, global_step):
    output_dir = os.path.join(sub_output_dir, f"checkpoint-{global_step}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info(f"AIMORE - Saving model checkpoint to {output_dir}")

def predict_on_website(per_gpu_eval_batch_size, n_gpu, local_rank, device, model, website):
    if website == "intralinks.com":
        print("Check", website)
    dataset, info = get_dataset_and_info_for_websites([website], evaluate=True)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    # In our setting, we should not apply DDP
    eval_sampler = SequentialSampler(dataset) if local_rank == -1 else DistributedSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running evaluation *****")
    logger.info(f"  Num examples for {website} = {len(dataset)}")
    logger.info(f"  Batch size = {eval_batch_size}")

    all_logits = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(b.to(device) for b in batch) # TODO (AIMORE): Why they use tuple here?
        with torch.no_grad():
            inputs = { # TODO (AIMORE): Can't this batch have better names, instead of these numbered indices?
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "xpath_tags_seq": batch[3],
                "xpath_subs_seq": batch[4],
            }
            outputs = model(**inputs)
            logits = outputs["logits"]  # which is (bs,seq_len,node_type)
            all_logits.append(logits.detach().cpu())

    all_probs = torch.softmax(torch.cat(all_logits, dim=0), dim=2)  # (all_samples, seq_len, node_type)

    assert len(all_probs) == len(info)
    #? info = It is a list of tuples, each tuple is: ('canelamedia.com.pick...e-0000.htm', [1, 1, 14, 14], ['/html/head', '/html/head/title', '/html/head/title/tail', '/html/head/script[1]'], ['none', 'none', 'none', 'none'], ['', 'Home - Canela Media ... Audiences', '', '{"@context":"https:/...ome"}}]}]}'])
    
    all_res = collections.defaultdict(dict)

    for sub_prob, sub_info in zip(all_probs, info):
        (
            html_path,
            involved_first_tokens_pos,
            involved_first_tokens_xpaths,
            involved_first_tokens_types,
            involved_first_tokens_text,
        ) = sub_info

        for pos, xpath, type, text in zip(
            involved_first_tokens_pos,
            involved_first_tokens_xpaths,
            involved_first_tokens_types,
            involved_first_tokens_text,
        ):

            pred = sub_prob[pos] #? This gets the first logit of each respective node
            #? sub_prob = [tensor([0.0045, 0.9955]), ...], sub_prob.shape = [384, 2]
            #? pos = 14
            #? pred = tensor([0.0045, 0.9955])
            if xpath not in all_res[html_path]:
                all_res[html_path][xpath] = {}
                all_res[html_path][xpath]["pred"] = pred
                all_res[html_path][xpath]["truth"] = type
                all_res[html_path][xpath]["text"] = text
            else:
                all_res[html_path][xpath]["pred"] += pred
                assert all_res[html_path][xpath]["truth"] == type
                assert all_res[html_path][xpath]["text"] == text

    lines = {
        "html_path": [],
        "xpath": [],
        "text": [],
        "truth": [],
        "pred_type": [],
        "final_probs": []
        }

    for html_path in all_res:
        # E.g. all_res [dict] = {html_path = {xpath = {'pred': tensor([0.4181, 0.5819]), 'truth': 'PAST_CLIENT', 'text': 'A healthcare client gains control of their ACA processes | BerryDunn'},...}, ...}
        for xpath in all_res[html_path]:
            final_probs = all_res[html_path][xpath]["pred"] / torch.sum(all_res[html_path][xpath]["pred"])  # TODO(aimore): Why is this even here? torch.sum(both prob) will always be 1, what is the point then? Maybe in case of more than one label?
            pred_id = torch.argmax(final_probs).item()
            pred_type = constants.ATTRIBUTES_PLUS_NONE[pred_id]
            final_probs = final_probs.numpy().tolist()

            lines["html_path"].append(html_path)
            lines["xpath"].append(xpath)
            lines["text"].append(all_res[html_path][xpath]["text"])
            lines["truth"].append(all_res[html_path][xpath]["truth"])
            lines["pred_type"].append(pred_type)
            lines["final_probs"].append(final_probs)

    result_df = pd.DataFrame(lines)
    return result_df 

def evaluate(per_gpu_eval_batch_size, n_gpu, local_rank, device, model, test_websites, **kwargs):
    """
    Evaluate the model
    """
    dataset_predicted = pd.DataFrame()
    for website in tqdm(test_websites):
        website_predicted = predict_on_website(per_gpu_eval_batch_size, n_gpu, local_rank, device, model, website)
        website_predicted['domain'] = website
        dataset_predicted = dataset_predicted.append(website_predicted)
    
    metrics_per_dataset, cm_per_dataset = compute_metrics_per_dataset(dataset_predicted)
    print(f"metrics_per_dataset: {metrics_per_dataset} | cm_per_dataset: {cm_per_dataset}")

    # compute_metrics_per_domain(dataset_predicted)    
    # compute_metrics_per_page(dataset_predicted)

    return metrics_per_dataset

def load_and_cache_one_website(arguments):
    args, tokenizer, website = arguments
    cached_features_file = os.path.join(
        args.root_dir,
        "cached",
        website,
        f"cached_markuplm_{str(args.max_seq_length)}",
    )

    if not os.path.exists(os.path.dirname(cached_features_file)):
        os.makedirs(os.path.dirname(cached_features_file))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file: {cached_features_file}")
        features = torch.load(cached_features_file)

    else:
        logger.info(f"Creating features for: {website}")

        features = get_swde_features(
            root_dir=args.root_dir,
            website=website,
            tokenizer=tokenizer,
            doc_stride=args.doc_stride,
            max_length=args.max_seq_length,
        )

        if args.local_rank in [-1, 0] and args.save_features:
            logger.info(f"Saving features into cached file: {cached_features_file}\n")
            torch.save(features, cached_features_file)

    return website, features


def load_and_cache_websites(args, tokenizer, websites:List):
    """
    Load and process the raw data.
    """
    if args.local_rank not in [-1, 0]:
       torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    feature_dicts = {}

    num_cores = mp.cpu_count()
    with mp.Pool(num_cores) as pool, tqdm(total=len(websites), desc="Loading/Creating Features") as t:
        arguments = zip([args]*len(websites), [tokenizer]*len(websites), websites)
        for website, features_per_website in pool.imap_unordered(load_and_cache_one_website, arguments):
            feature_dicts[website] = features_per_website

    # for website in websites:
    #     features_per_website = load_and_cache_one_website(args, tokenizer, website)
    #     feature_dicts[website] = features_per_website

    return feature_dicts


def get_dataset_and_info_for_websites(websites: List, evaluate=False):
    """
    Args:
        websites: a list of websites
    Returns:
        a dataset object and info
    """

    all_features = []

    for website in websites:
        features_per_website = global_feature_dicts[website]
        all_features += features_per_website

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in all_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in all_features], dtype=torch.long)
    all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in all_features], dtype=torch.long)
    all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in all_features], dtype=torch.long)

    if not evaluate:
        all_labels = torch.tensor([f.labels for f in all_features], dtype=torch.long)
        dataset = SwdeDataset(
            all_input_ids=all_input_ids,
            all_attention_mask=all_attention_mask,
            all_token_type_ids=all_token_type_ids,
            all_xpath_tags_seq=all_xpath_tags_seq,
            all_xpath_subs_seq=all_xpath_subs_seq,
            all_labels=all_labels,
        )
        info = None
    else:
        # in evaluation, we do not add labels
        dataset = SwdeDataset(
            all_input_ids=all_input_ids,
            all_attention_mask=all_attention_mask, 
            all_token_type_ids=all_token_type_ids,
            all_xpath_tags_seq=all_xpath_tags_seq,
            all_xpath_subs_seq=all_xpath_subs_seq,
        )
        info = [
            (
                f.html_path, #? '1820productions.com.pickle-0000.htm'
                f.involved_first_tokens_pos, #? [1, 1, 34, 70, 80]
                f.involved_first_tokens_xpaths, #? ['/html/head', '/html/head/script[1]', '/html/head/script[2]', '/html/head/title', '/html/head/script[3]']
                f.involved_first_tokens_types, #? ['none', 'none', 'none', 'none', 'none']
                f.involved_first_tokens_text, #? ['', "var siteConf = { ajax_url: 'https://1820productions.com/wp-admin/admin-ajax.php' };", "(function(html){html.className = html.className.replace(/\\bno-js\\b/,'js')})(document.documentElement);", 'Portfolio | 1820 Productions - Video Production Dallas', 'window._wpemojiSettings = {"baseUrl":"https:\\/\\/s.w.org\\/images\\/core\\/emoji\\/11\\/72x72\\/","ext":".png","svgUrl":"https:\\/\\/s.w.org\\/images\\/core\\/emoji\\/11\\/svg\\/","svgExt":".svg","source":{"concatemoji":"https:\\/\\/1820productions.com\\/wp-includes\\/js\\/wp-emoji-release.min.js?ver=4.9.18"}};!function(e,a,t){var n,r,o,i=a.createElement("canvas"),p=i.getContext&&i.getContext("2d");function s(e,t){var a=String.fromCharCode;p.clearRect(0,0,i.width,i.height),p.fillText(a.apply(this,e),0,0);e=i.toDataURL();return p.clearRect(0,0,i.width,i.height),p.fillText(a.apply(this,t),0,0),e===i.toDataURL()}function c(e){var t=a.createElement("script");t.src=e,t.defer=t.type="text/javascript",a.getElementsByTagName("head")[0].appendChild(t)}for(o=Array("flag","emoji"),t.supports={everything:!0,everythingExceptFlag:!0},r=0;r<o.length;r++)t.supports[o[r]]=function(e){if(!p||!p.fillText)return!1;switch(p.textBaseline="top",p.font="600 32px Arial",e){case"flag":return s([55356,56826,55356,56819],[55356,56826,8203,55356,56819])?!1:!s([55356,57332,56128,56423,56128,56418,56128,56421,56128,56430,56128,56423,56128,56447],[55356,57332,8203,56128,56423,8203,56128,56418,8203,56128,56421,8203,56128,56430,8203,56128,56423,8203,56128,56447]);case"emoji":return!s([55358,56760,9792,65039],[55358,56760,8203,9792,65039])}return!1}(o[r]),t.supports.everything=t.supports.everything&&t.supports[o[r]],"flag"!==o[r]&&(t.supports.everythingExceptFlag=t.supports.everythingExceptFlag&&t.supports[o[r]]);t.supports.everythingExceptFlag=t.supports.everythingExceptFlag&&!t.supports.flag,t.DOMReady=!1,t.readyCallback=function(){t.DOMReady=!0},t.supports.everything||(n=function(){t.readyCallback()},a.addEventListener?(a.addEventListener("DOMContentLoaded",n,!1),e.addEventListener("load",n,!1)):(e.attachEvent("onload",n),a.attachEvent("onreadystatechange",function(){"complete"===a.readyState&&t.readyCallback()})),(n=t.source||{}).concatemoji?c(n.concatemoji):n.wpemoji&&n.twemoji&&(c(n.twemoji),c(n.wpemoji)))}(window,document,window._wpemojiSettings);']
            )
            for f in all_features
        ]

    return dataset, info #? This info is used for for evaluation (store the groundtruth) 


def save_model_and_tokenizer(sub_output_dir, model, tokenizer, args):
    # Create output directory if needed
    if not os.path.exists(sub_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(sub_output_dir)

    logger.info(f"Saving model checkpoint to {sub_output_dir}")
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(sub_output_dir)
    tokenizer.save_pretrained(sub_output_dir)
    torch.save(args, os.path.join(sub_output_dir, "training_args.bin"))

def do_something(train_websites, test_websites, args, config, tokenizer):
    # before each run, we reset the seed
    set_seed(args.seed, args.n_gpu)

    model = MarkupLMForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # My modification
    sub_output_dir = os.path.join(
        args.output_dir,
        str(len(train_websites))
    )

    # if args.local_rank == 0:
    #     torch.distributed.barrier()
    # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info(f"Training/evaluation parameters {args}")

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is
    # set. Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running
    # `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
    print(f"sub_output_dir:{sub_output_dir}")  # TODO (aimore): perhaps change this name.
    # Training
    if args.do_train:
        train_dataset, _ = get_dataset_and_info_for_websites(train_websites)
        tokenizer.save_pretrained(sub_output_dir)
        model.to(args.device)
        global_step, tr_loss = train(**args.__dict__, train_dataset=train_dataset, model=model, sub_output_dir=sub_output_dir)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        save_model_and_tokenizer(sub_output_dir, model, tokenizer, args)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info(f"Evaluate the following model: {args.model_path}")
        config = MarkupLMConfig.from_pretrained(args.model_path)
        tokenizer = MarkupLMTokenizer.from_pretrained(args.model_path)

        model = MarkupLMForTokenClassification.from_pretrained(args.model_path, config=config)
        model.to(args.device)

        result = evaluate(**args.__dict__, model=model, test_websites=test_websites)

        logger.info(f"Results: {result}")

        return result


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--root_dir",
        default=None,
        type=str,
        required=True,
        help="the root directory of the pre-processed SWDE dataset, "
        "in which we have `book-abebooks-2000.pickle` files like that",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_eval", 
        action="store_true", 
        help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending"
        "with step number",
    )
    parser.add_argument(
        "--eval_from_checkpoint",
        type=int,
        default=0,
        help="Only evaluate the checkpoints with prefix larger than or equal to it, beside the final"
        "checkpoint with no prefix",
    )
    parser.add_argument(
        "--eval_to_checkpoint",
        type=int,
        default=None,
        help="Only evaluate the checkpoints with prefix smaller than it, beside the final checkpoint"
        "with no prefix",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.0,
        type=float,
        help="Linear warmup ratio over all steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=3000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether not to use CUDA when available",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--save_features",
        type=bool,
        default=True,
        help="whether or not to save the processed features, default is True",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="NO MODEL",
        help="The path of the model to be used for evaluation.",
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.fp16}"
    )

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    config = MarkupLMConfig.from_pretrained(args.model_name_or_path)
    config_dict = config.to_dict()
    config_dict.update({"node_type_size": len(constants.ATTRIBUTES_PLUS_NONE)})
    config = MarkupLMConfig.from_dict(config_dict)

    tokenizer = MarkupLMTokenizer.from_pretrained(args.model_name_or_path)

    swde_path = Path(args.root_dir)
    websites = [
        file_path.parts[-1]
        for file_path in list(swde_path.iterdir())
        if "cached" not in str(file_path)
    ]

    websites = [website for website in websites if "ciphr.com" not in website] #! Remove this website for now just because it is taking too long (+20min.) 
    # websites = websites[:10] #! Just for speed reasons

    train_websites = websites
    test_websites = websites

    logger.info(f"\nWebsites ({len(websites)}):\n{websites}\n")

    # first we load the features
    feature_dicts = load_and_cache_websites(
        args=args,
        tokenizer=tokenizer,
        websites=websites,
    )

    global global_feature_dicts
    global_feature_dicts = feature_dicts

    eval_res = do_something(train_websites, test_websites, args, config, tokenizer)
    
    if eval_res:
        all_precision = eval_res["precision"]
        all_recall = eval_res["recall"]
        all_f1 = eval_res["f1"]

        logger.info("=================FINAL RESULTS=================")
        logger.info(f"Precision : {all_precision}")
        logger.info(f"Recall : {all_recall}")
        logger.info(f"F1 : {all_f1}")

        res_file = os.path.join(args.output_dir, f"run-score.txt")

        with open(res_file, "w") as fio:
            fio.write(f"Precision : {all_precision}\nRecall : {all_recall}\nF1 : {all_f1}\n")


if __name__ == "__main__":
    main()
