import torch
from typing import Optional, Sequence, Tuple
from transformers import AutoTokenizer, RobertaTokenizer
from urllib.parse import unquote, urlparse
import re
from lxml import etree
import lxml
from lxml.html.clean import Cleaner
import unicodedata
from markuplmft.fine_tuning.run_swde import constants
from markuplmft.data.tag_utils import tags_dict
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class WindowFeature(object):
    def __init__(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        url,
        node_ids,
        labels=None,
    ):
        self.input_ids = input_ids
        # node_ids_padded = np.pad(
        #     np.array(node_ids), (0, len(input_ids) - len(node_ids)), "constant", constant_values=0
        # )
        # self.token_type_ids = node_ids_padded
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels

        self.url = url
        self.node_ids = node_ids


class SwdeDataset(Dataset):
    def __init__(
        self,
        model_input,
        urls,
        node_ids,
    ):
        self.model_input = model_input
        self.urls = urls
        self.node_ids = node_ids

    def __len__(self):
        return len(self.model_input["labels"])

    def __getitem__(self, index):
        return {
            key: tensor[index]
            for key, tensor in self.model_input.items()
        }  # type: ignore


class Featurizer:
    def __init__(
        self,
        doc_stride=128,
        max_length=384,
        tokenizer=AutoTokenizer.from_pretrained("roberta-base", use_fast=True),
    ) -> None:
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_length = max_length

    def feature_to_dataset(self, features: Sequence[WindowFeature]) -> SwdeDataset:
        input_ids_tensor = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask_tensor = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        token_type_ids_tensor = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        labels_tensor = torch.tensor([f.labels for f in features], dtype=torch.long)
        print(
            f"Label distribution: {pd.Series(labels_tensor.reshape(-1, 1).squeeze()).value_counts()}"
        )
        for x in list(pd.Series(labels_tensor.reshape(-1, 1).squeeze()).value_counts().index):
            assert x in [0, 1, -100]

        model_input = dict(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            token_type_ids=token_type_ids_tensor,
            labels=labels_tensor,
        )
        urls = [f.url for f in features]
        node_ids = [f.node_ids for f in features]

        return SwdeDataset(
            model_input,
            urls,
            node_ids,
        )

    def get_page_features(
        self, url: str, nodes: Sequence[Tuple[str, str, str, Sequence[str]]]
    ) -> Sequence[WindowFeature]:
        """
        nodes: [(xpath, node_text, node_tag, gt_text), ...]
        1. Tokenizer node texts
        2. Convert tokens into ids
        3. Stride over all tokens ids and create a 384 token window view feature
        4. Move window by a doc_stride of 128
        """
        # TODO: Move this to a function called tokenize_all_nodes_in_a_page
        # TODO: Deal when there is no tag
        nodes_df = pd.DataFrame(nodes, columns=["xpath", "node_text", "node_tag", "node_gt_text"])

        nodes_df["node_text_tok_ids"] = nodes_df["node_text"].apply(
            lambda row: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(row))
        )  # TODO (speed up): Make use of tokenize batches

        nodes_df["node_text_tok_ids_padded"] = nodes_df["node_text_tok_ids"].apply(
            lambda row: row + [self.tokenizer.pad_token_id]  # TODO: Try cls token
        )
        nodes_df["node_text_tok_ids_padded_labels_seq"] = nodes_df.apply(
            lambda row: [constants.ATTRIBUTES_PLUS_NONE.index(row["node_tag"])]
            * len(row["node_text_tok_ids"])
            + [-100],
            axis=1,
        )

        tokens_df = nodes_df.explode(
            ["node_text_tok_ids_padded", "node_text_tok_ids_padded_labels_seq"]
        )
        tokens_df.reset_index(inplace=True)
        tokens_df = tokens_df.rename({"index": "node_ids"}, axis=1)
        last_token = len(tokens_df)

        page_features = []
        for window in range(0, last_token, self.doc_stride):
            feat = tokens_df[
                [
                    "node_ids",
                    "node_text_tok_ids_padded",
                    "node_text_tok_ids_padded_labels_seq",
                ]
            ].loc[window : (window + self.max_length - 3)]

            splited_page_tokens_ids = (
                [self.tokenizer.cls_token_id]
                + feat["node_text_tok_ids_padded"].to_list()
                + [self.tokenizer.sep_token_id]
            )

            current_len = len(splited_page_tokens_ids)
            token_type_ids = [0] * self.max_length
            attention_mask = [1] * current_len
            splited_labels_seq = (
                [-100] + feat["node_text_tok_ids_padded_labels_seq"].to_list() + [-100]
            )
            node_ids = feat["node_ids"].values
            # ?Pad
            # input_ids, attention_mask = self.tokenizer.prepare_for_model(feat["node_text_tok_ids_padded"].to_list())[["input_ids", "attention_mask"]]
            if current_len < self.max_length:
                tokens_left = self.max_length - current_len
                splited_page_tokens_ids += [self.tokenizer.pad_token_id] * tokens_left
                splited_labels_seq += [-100] * tokens_left
                attention_mask += [0] * tokens_left

            page_features.append(
                WindowFeature(
                    input_ids=splited_page_tokens_ids,  # ? len = 384,
                    token_type_ids=token_type_ids,  # ? Always 0. len = 384,
                    attention_mask=attention_mask,  # ? Always 1. len = 384,
                    url=url,
                    node_ids=node_ids,
                    labels=splited_labels_seq,
                )
            )
        return page_features

    def get_domain_from_url(self, url: str) -> str:
        """
        "http://en.added-value.com/case-study/career-educa.php" --> "en.added-value.com"
        """
        url = unquote(url).lower().strip("/")
        url_parsed = urlparse(url)
        domain = url_parsed.netloc
        return domain

    def clean_the_url(self, url: str) -> str:
        "Remove domain and symbols from the url"
        domain = self.get_domain_from_url(url)
        url_without_domain = url.split(domain)[1]
        clean_url = re.sub("[%+\./:?-]", " ", url_without_domain)  # ? Replace symbols with spaces
        clean_url = re.sub("\s+", " ", clean_url)  # ? Reduce any space to one space
        return clean_url

    def insert_url_into_html(self, url: str, html: str) -> str:
        "Clean url and add to html as a title node"
        clean_url = self.clean_the_url(url)
        element = etree.Element("title")
        element.text = f" {clean_url} "
        dom_tree = get_dom_tree(html)
        root = dom_tree.getroot()
        root.insert(0, element)
        etree.indent(dom_tree)
        etree.tostring(dom_tree, encoding=str)
        return html

    def clean_annotations(self):
        # ! This is not implemented yet, but I think it would be good to have a standard way of cleaning annotations
        # ! Not done yet because I need to think how that will interact with finding it in the page.
        pass

    def get_nodes(self, html: str) -> Optional[Sequence[Tuple[str, str]]]:
        """
        Get important nodes and their important attributes as a tuple.

        return (xpath, text, node_tag, gt_texts) or None
        """
        # dom_tree = etree.ElementTree(lxml.html.fromstring(html)) #! Faster but not correct use below
        dom_tree = get_dom_tree(html)

        page_nodes = []
        min_node_text_size, max_node_text_size = 2, 10_000

        for node in dom_tree.iter():
            node_text_dict = {
                "node_text": node.text,
                "node_tail_text": node.tail,
            }
            for text_part_flag, node_text in node_text_dict.items():
                if node_text:
                    if (
                        node.tag != "script"
                        and "javascript" not in node.attrib.get("type", "")
                        and min_node_text_size <= len(node_text.strip()) < max_node_text_size
                    ):  #! Remove java/script and min_node_text # TODO (Aimore): Make this comparisons more explicity and descriptive

                        # node_attribute = node.attrib.get("type", "")
                        # node_tag = node.tag

                        node_text_split = node_text.split("--BRRB--")
                        len_brs = len(node_text_split)  # The number of the <br>s.

                        for index, etext in enumerate(node_text_split):
                            if text_part_flag == "node_text":
                                xpath = dom_tree.getpath(node)

                            elif text_part_flag == "node_tail_text":
                                xpath = dom_tree.getpath(node) + "/tail"

                            if len_brs >= 2:
                                xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]

                            page_nodes.append((xpath, etext, "none", []))
        # ? Make sure that it is returning a page with features
        if len(page_nodes) > 0:
            return page_nodes
        else:
            print(f"No nodes from this html were able to be extracted - html: {html}")


def _clean_spaces(text: str) -> str:
    r"""Clean extra spaces in a string.

    Example:
    input: " asd  qwe   " --> output: "asd qwe"
    input: " asd\t qwe   " --> output: "asd qwe"
    Args:
    text: the input string with potentially extra spaces.

    Returns:
    a string containing only the necessary spaces.
    """
    return " ".join(re.split(r"\s+", text.strip()))


def _clean_format_str(text: str) -> str:
    """Cleans unicode control symbols, non-ascii chars, and extra blanks."""
    text = "".join(
        ch
        for ch in text
        if unicodedata.category(ch)[0] != "C"
    )
    text = "".join([
        c if ord(c) < 128 else ""
        for c in text
    ])
    text = _clean_spaces(text)
    return text


def get_dom_tree(html: str) -> etree.ElementTree:
    cleaner = Cleaner()
    cleaner.javascript = True
    cleaner.scripts = True
    cleaner.style = True
    cleaner.page_structure = False

    html = html.replace("\0", "")  # ? Delete NULL bytes.
    # ? Replace the <br> tags with a special token for post-processing the xpaths.
    html = html.replace("<br>", "--BRRB--")
    html = html.replace("<br/>", "--BRRB--")
    html = html.replace("<br />", "--BRRB--")
    html = html.replace("<BR>", "--BRRB--")
    html = html.replace("<BR/>", "--BRRB--")
    html = html.replace("<BR />", "--BRRB--")

    html = _clean_format_str(html)
    # TODO(Aimore): Deal with XML cases. If there are problems here with XLM, is because it can only treat HTMLpages

    html = lxml.html.fromstring(html)
    etree_root = cleaner.clean_html(html)
    dom_tree = etree.ElementTree(etree_root)
    return dom_tree
