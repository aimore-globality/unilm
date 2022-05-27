import pandas as pd
import pandavro as pdx
import re
from ast import literal_eval
from pathlib import Path
from lxml import html as lxml_html
import wandb
import lxml
import os
import shutil
import os
from microcosm.api import create_object_graph
import os
import re
import unicodedata
from lxml import etree
from lxml.html.clean import Cleaner
import multiprocessing as mp
from pathlib import Path
import os
from pathlib import Path
from lxml.html.clean import Cleaner
import unicodedata
from urllib.parse import unquote, urlparse


def clean_spaces(text):
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


def clean_format_str(text):
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
    text = clean_spaces(text)
    return text


def get_dom_tree(html):
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

    html = clean_format_str(html)
    # TODO(Aimore): Deal with XML cases. If there are problems here with XLM, is because it can only treat HTMLpages

    html = lxml.html.fromstring(html)
    etree_root = cleaner.clean_html(html)
    dom_tree = etree.ElementTree(etree_root)
    return dom_tree


def get_annotations(annotations: pd.Series, annotation_name: str):
    return annotations.apply(
        lambda annotations: [
            annotation[annotation_name]
            for annotation in annotations
            if annotation[annotation_name]
        ]
    )


# ? Create mapping to convert gt_value_taxonomy into gt_value
graph = create_object_graph("test")
taxonomy_to_value_mappings = dict(
    [
        (company.uri, company.name)
        for company in graph.known_company_taxonomy
    ]
)


def untaxonomize_gt_value(gt_value: str):
    gt_value_untax = taxonomy_to_value_mappings.get(gt_value)
    return gt_value_untax


import tqdm
from torch.utils.data import Dataset
from markuplmft.data.tag_utils import tags_dict
import pickle
import os
from markuplmft.fine_tuning.run_swde import constants


class SwdeFeature(object):  # BatchEncoding PageClassifierFeature
    def __init__(
        self,
        html_path,
        input_ids,
        token_type_ids,
        attention_mask,
        labels,
        involved_first_tokens_pos,
        involved_first_tokens_types,
        involved_first_tokens_text,
        involved_first_tokens_gt_text,
        involved_first_tokens_node_attribute,
        involved_first_tokens_node_tag,
    ):
        """
        html_path: indicate which page the feature belongs to
        input_ids: RT
        token_type_ids: RT
        attention_mask: RT
        labels: RT
        involved_first_tokens_pos: a list, indicate the positions of the first-tokens in this feature
        involved_first_tokens_types: the types of the first-tokens
        involved_first_tokens_text: the text of the first tokens

        Note that `involved_xxx` are not fixed-length array, so they shouldn't be sent into our model
        They are just used for evaluation
        """
        self.html_path = html_path
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.involved_first_tokens_pos = involved_first_tokens_pos
        self.involved_first_tokens_types = involved_first_tokens_types
        self.involved_first_tokens_text = involved_first_tokens_text
        self.involved_first_tokens_gt_text = involved_first_tokens_gt_text
        self.involved_first_tokens_node_attribute = involved_first_tokens_node_attribute
        self.involved_first_tokens_node_tag = involved_first_tokens_node_tag


class SwdeDataset(Dataset):
    def __init__(
        self,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels=None,
    ):

        self.tensors = [
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
        ]

        if not all_labels is None:
            self.tensors.append(all_labels)

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


class Featurizer:
    def __init__(
        self,
        tokenizer,
        doc_stride,
        max_length,
    ) -> None:
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.max_length = max_length

    def get_domain_url(self, url: str) -> str:
        """
        "http://en.added-value.com/case-study/career-educa.php" --> "en.added-value.com"
        """
        url = unquote(url).lower().strip("/")
        url_parsed = urlparse(url)
        return url_parsed.netloc

    def clean_the_url(self, url):
        "Remove domain and symbols from the url"
        domain = self.get_domain_url(url)
        url_without_domain = url.split(domain)[1]
        clean_url = re.sub("[%+\./:?-]", " ", url_without_domain)  # ? Replace symbols with spaces
        clean_url = re.sub("\s+", " ", clean_url)  # ? Reduce any space to one space
        return clean_url

    def insert_url_into_html(self, url, html):
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

    def get_nodes(self, html):
        """
        Get important nodes and their important attributes as a tuple.

        return (text, xpath) or None
        """
        dom_tree = etree.ElementTree(lxml.html.fromstring(html))

        page_features = []
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

                            page_features.append((xpath, etext))

        return page_features

    def get_swde_features(self, nodes):
        """
        raw_data: [(text, xpath, tag, gt_text)]
        This function creates a list of features that goes into the model.
        The data already comes divided by nodes.
        From the remaining nodes, features are created.
        Each feature has 384 tokens.
        Each feature represents a doc_stride

        """
        real_max_token_num = self.max_length - 2  # for cls and sep

        features = []

        all_tokens = []
        token_to_ori_map_seq = []
        all_labels_seq = []

        first_token_pos = []
        first_token_xpaths = []
        first_token_type = []
        first_token_text = []
        first_token_gt_text = []
        first_token_node_attribute = []
        first_token_node_tag = []

        # This for loop goes over the selected nodes and append the tokens and xpaths from each node
        for i, node in enumerate(nodes):
            node_text, xpath, tag, gt_text = node[0], node[1], node[2], node[3]
            # ? E.g. text [str] = 'HITT FUTURES'
            # ? E.g. xpath [str] = '/html/body/div/div/div[2]/div[1]/div[2]/div/div/ul/li[3]/a'
            # ? E.g. type [str] = 'fixed-node'
            token_ids = self.tokenizer.tokenize(node_text)
            all_tokens += token_ids

            token_to_ori_map_seq += [i] * len(token_ids)

            # we always use the first token to predict
            first_token_pos.append(len(all_labels_seq))  # ? E.g. len(all_labels_seq) = 71
            # ? E.g. first_token_pos = [71, 95, 100, 104, 184, 192, 198, 212]
            first_token_type.append(type)  # ? E.g. type = 'none'
            # ? E.g. first_token_type = ['none', 'none', 'none', 'none', 'PAST_CLIENT', 'none', 'none', 'none']
            first_token_xpaths.append(
                xpath
            )  # ? E.g. xpath = '/html/body/div/div/div[2]/div[1]/div[2]/div/div/ul/li[3]/a'
            first_token_text.append(text)  # ? E.g. text = 'HITT FUTURES'
            # ? E.g. ['1730 Pennsylvania Avenue NW | HITT', '1730 Pennsylvania Avenue NW', 'Washington, DC', "HITT completed an occupied building renovation of the main lobby, elevator cabsand typical tenant lobbies on four of the floors in this commercial office building loca

            first_token_gt_text.append(gt_text)
            first_token_node_attribute.append(node_attribute)
            first_token_node_tag.append(node_tag)

            all_labels_seq += [constants.ATTRIBUTES_PLUS_NONE.index(type)] * len(token_ids)
            # E. g. all_labels_seq = [-100, -100, ..., 1, 1, 1, 1, 1, -100, ..., -100, -100]
            # The numbers in each token_ids indicates the label index in constants.ATTRIBUTES_PLUS_NONE
            # This means that all tokens_ids for the text in the xpath
            # will get labelled as something differently than -100 in case it is positive.

            assert len(all_token_ids_seq) == len(all_xpath_tags_seq)
            assert len(all_token_ids_seq) == len(all_labels_seq)

            # we have all the pos of variable nodes in all_token_ids_seq
            # now we need to assign them into each feature

            start_pos = 0
            flag = False

            curr_first_token_index = 0

            # TODO (Aimore): Check if the nodes are being dropped somehow. It seems there are less nodes than it should be?
            while True:
                # ? This loop goes over all_token_ids_seq in a stride manner.
                # ? The first step is to get the features for the window.
                # ? invloved is [ start_pos , end_pos )

                token_type_ids = [0] * self.max_length  # that is always this

                end_pos = start_pos + real_max_token_num
                # add start_pos ~ end_pos as a feature
                splited_token_ids_seq = (
                    [tokenizer.cls_token_id]
                    + all_tokens[start_pos:end_pos]
                    + [tokenizer.sep_token_id]
                )
                # ? tokenizer.cls_token_id = 0
                # ? tokenizer.sep_token_id = 2
                # ? Gets a subset of the all_token_ids_seq and appends cls_token_id(0) to beginning and cls_token_id(2) to end.
                # ? The length of the subset is given by the real_max_token_num 382.
                # ? E.g. splited_token_ids_seq [len(382)] = [42996, 4, 23687, 48159, 5457, 2931, ...]

                splited_xpath_tags_seq = (
                    [padded_xpath_tags_seq]
                    + all_xpath_tags_seq[start_pos:end_pos]
                    + [padded_xpath_tags_seq]
                )
                splited_xpath_subs_seq = [padded_xpath_subs_seq] + [padded_xpath_subs_seq]
                splited_labels_seq = [-100] + all_labels_seq[start_pos:end_pos] + [-100]

                # ? locate first-tokens in them
                involved_first_tokens_pos = []
                involved_first_tokens_text = []

                while (
                    curr_first_token_index < len(first_token_pos)
                    and end_pos > first_token_pos[curr_first_token_index] >= start_pos
                ):  # ? This while doesn't run if first_token_pos[curr_first_token_index] is very high (above 382)
                    # ? This while loops over the first_token_pos and breaks if first_token_pos is higher than the end_pos

                    # ? +1 for [cls]
                    involved_first_tokens_pos.append(
                        first_token_pos[curr_first_token_index] - start_pos + 1
                    )
                    involved_first_tokens_text.append(first_token_text[curr_first_token_index])

                    curr_first_token_index += 1

                if end_pos >= len(all_tokens):
                    # ? This will be the last time of this loop.
                    flag = True
                    # ? which means we need to pad in this feature
                    current_len = len(splited_token_ids_seq)
                    splited_token_ids_seq += [tokenizer.pad_token_id] * (
                        self.max_length - current_len
                    )
                    splited_labels_seq += [-100] * (self.max_length - current_len)
                    attention_mask = [1] * current_len + [0] * (self.max_length - current_len)

                else:
                    # ? no need to pad, the splited seq is exactly with the length `self.max_length`
                    assert len(splited_token_ids_seq) == self.max_length
                    attention_mask = [1] * self.max_length

                features.append(
                    SwdeFeature(  # TODO (Aimore): If you put a breakpoint here you will see that many features are being created with empty text  -verify why
                        input_ids=splited_token_ids_seq,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=splited_labels_seq,
                        involved_first_tokens=involved_first_tokens,
                    )
                )
                start_pos += self.doc_stride

                if flag:
                    break

        return features
        # ? features = [swde_feature_1, swde_feature_2, ...]


class PrepareData:
    """
    - Convert CF data into SWDE format
    - Create the labels
    - Remove some data
    -
    """

    def __init__(self, tag="PAST_CLIENT"):
        self.tag = tag

    def load_data(self, load_data_path, limit=-1):
        print("load_data")
        self.wae_data_load_path = load_data_path
        df = pdx.read_avro(str(load_data_path / "dataset.avro"))
        df = df[:limit]
        df.annotations = df.annotations.apply(literal_eval)

        for column in ["url", "domain", "annotations"]:
            assert column in df.columns, f"Column: {column} not in DF"

        print(len(df))
        print("done")
        return df

    def format_annotation(self, df):
        print("Format_annotation...")
        df[f"{self.tag}-annotations"] = df["annotations"].apply(
            lambda annotation: annotation.get(self.tag)
        )
        df[f"{self.tag}-annotations"] = df[f"{self.tag}-annotations"].fillna("").apply(list)

        df[f"{self.tag}-gt_text"] = get_annotations(df[f"{self.tag}-annotations"], "text")
        df[f"{self.tag}-gt_value"] = get_annotations(df[f"{self.tag}-annotations"], "value")
        df[f"{self.tag}-gt_value_untax"] = df[f"{self.tag}-gt_value"].apply(
            lambda gt_values: [untaxonomize_gt_value(gt_value) for gt_value in gt_values]
        )
        df[f"{self.tag}-annotations-untax"] = df[f"{self.tag}-annotations"].apply(
            lambda annotations: [
                {
                    "gt_text": annotation["text"],
                    "gt_value_untax": untaxonomize_gt_value(annotation["value"]),
                }
                for annotation in annotations
            ]
        )
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len)

        print(f" # Annotations (gt_text): {self.count_all_labels(df)}")
        print(f" # Annotations (gt_value): {self.count_all_labels(df, 'value')}")
        return df

    def create_postive_negative_data(self, df, negative_fraction=0.1):
        print("Convert_annotated_data...")
        df_positives, df_negatives, df_negatives_sample = self.get_negative_fraction(
            df, negative_fraction
        )
        print("- Positives:")
        df_positives = self.remove_non_html_pages(
            df_positives
        )  # TODO(Aimore): Try to move this out
        print("- Negatives:")
        df_negatives_sample = self.remove_non_html_pages(
            df_negatives_sample
        )  # TODO(Aimore): Try to move this out

        df_positives = self.remove_annotations_from_images(df_positives)
        df_positives = self.remove_annotations_that_cannot_be_found_on_html(df_positives)

        # ? From df_negatives_sample filter out domains that are not in df_positives
        df_negatives_sample = df_negatives_sample[
            df_negatives_sample["domain"].isin(df_positives["domain"])
        ]
        print(
            f"Positive Domains: {len(set(df_positives['domain']))} | Negative Domains: {len(set(df_negatives_sample['domain']))}"
        )
        assert (
            len(set(df_negatives_sample["domain"]) - set(df_positives["domain"])) == 0
        ), "Negatives have a domain that positive doesnt have!"

        # ? Make sure that the ratio is still the same
        df_negatives_positive_domain = df_negatives[
            df_negatives["domain"].isin(df_positives["domain"])
        ]
        final_negative_fraction = len(df_negatives_sample) / len(df_negatives_positive_domain)
        print(
            f" # of Pages (Negative Sample): {len(df_negatives_sample)} ({100*final_negative_fraction:.4f} %) \n # of Pages (Negative): {len(df_negatives_positive_domain)}"
        )
        assert negative_fraction - 0.01 < final_negative_fraction < negative_fraction + 0.01

        # ? Merge positives and negatives
        df_positives_negatives = df_positives.append(df_negatives_sample)
        print(
            f"# Total Pages (positive and negatives): {len(df_positives_negatives)} \n Total Domains: {len(set(df_positives_negatives['domain']))}"
        )

        # ? Save this dataset that is used to compare with production
        save_intermediate_path = (
            self.wae_data_load_path
            / f"dataset_pos({len(df_positives)})_neg({len(df_negatives_sample)})_intermediate.pkl"
        )
        print(f"Saving file: {save_intermediate_path}")
        df_positives_negatives.to_pickle(save_intermediate_path)

        # ? Check the amount of annotations in each domain
        print(
            pd.DataFrame(
                df_positives_negatives.groupby("domain").sum("PAST_CLIENT-gt_text_count")
            ).sort_values("PAST_CLIENT-gt_text_count", ascending=False)
        )
        print("done")
        return df_positives_negatives

    def get_negative_fraction(self, df, negative_fraction=0.10):
        print("Get_negative_fraction...")
        df_positives = df[df[f"{self.tag}-gt_text_count"] > 0]
        df_negatives = df[df[f"{self.tag}-gt_text_count"] == 0]

        df_negatives_sample = df_negatives

        # domains_20_or_less = (
        #     df_negatives.groupby("domain")["url"]
        #     .count()[df_negatives.groupby("domain")["url"].count() <= 20]
        #     .index
        # )
        # domains_more_than_20 = (
        #     df_negatives.groupby("domain")["url"]
        #     .count()[df_negatives.groupby("domain")["url"].count() > 20]
        #     .index
        # )

        # df_negatives_sample = (
        #     df_negatives[df_negatives["domain"].isin(domains_more_than_20)]
        #     .groupby("domain")
        #     .sample(frac=negative_fraction, random_state=66)
        # )
        # df_negatives_sample = df_negatives_sample.append(
        #     df_negatives[df_negatives["domain"].isin(domains_20_or_less)]
        # )

        print(
            f"# Pages:\nNegatives: {len(df_negatives)} | Negatives sample: {len(df_negatives_sample)} | Positives:{len(df_positives)}"
        )
        return df_positives, df_negatives, df_negatives_sample

    def count_all_labels(self, df, tag_type="text"):
        return df[f"{self.tag}-gt_{tag_type}"].apply(len).sum()

    def remove_non_html_pages(self, df):
        pages_without_html_explicity = df[df["html"] == "PLACEHOLDER_HTML"]
        print(f"# of Pages that are not html explicity: {len(pages_without_html_explicity)}")
        print(
            f"# of Annotations (gt_text) that are not html explicity: {self.count_all_labels(pages_without_html_explicity)}"
        )
        df = df[df["html"] != "PLACEHOLDER_HTML"]

        def get_only_html(t):
            """Deal with XLM cases"""
            text = "NOT HTML"
            try:
                text = lxml_html.fromstring(t)
                return t
            except:
                return text

        pages_with_html = df["html"].apply(get_only_html)
        pages_without_html_implicity = df[pages_with_html == "NOT HTML"]
        print(f"# of Pages that are not html implicity: {len(pages_without_html_implicity)}")
        print(
            f"# of Annotations (gt_text) that are not html implicity: {self.count_all_labels(pages_without_html_implicity)}"
        )
        df = df[pages_with_html != "NOT HTML"]

        return df

    def remove_annotations_from_images(self, df):
        print("remove_annotations_from_images")
        print(f"# of Annotations (gt_text) before: {self.count_all_labels(df)}")
        df[f"{self.tag}-gt_text"] = df[f"{self.tag}-gt_text"].apply(
            lambda annotations: [
                annotation
                for annotation in annotations
                if "htt" not in annotation
            ]
        )
        print(f"# of Annotations (gt_text) after: {self.count_all_labels(df)}")
        print("done")
        return df

    def remove_annotations_that_cannot_be_found_on_html(self, df):
        print("remove_annotations_that_cannot_be_found_on_html")
        initial_amount_of_label = self.count_all_labels(df)

        all_annotations_left = []

        for enum, row in df.iterrows():
            url = row["url"]
            if not row.isnull()[f"{self.tag}-gt_text"]:
                # clean_dom_tree = get_dom_tree(row["html"])
                # dom_tree = clean_dom_tree
                dom_tree = lxml_html.fromstring(
                    row["html"]
                )  #! The line above is slower, but that is what it is done when creating the html which the model will see

                annotations_that_can_be_found = []
                annotations_that_cannot_be_found = []
                for text_annotation in row[f"{self.tag}-gt_text"]:
                    found = False
                    # TODO (Aimore): This process is very similar to the one that actually annotates the nodes. It would be better if they were reused.
                    for node in dom_tree.iter():
                        if node.text:
                            if text_annotation.lower() in node.text.lower():
                                annotations_that_can_be_found.append(text_annotation)
                                found = True
                                break
                        if node.tail:
                            if text_annotation.lower() in node.tail.lower():
                                annotations_that_can_be_found.append(text_annotation)
                                found = True
                                break
                        # #? In case I want to add the images:
                        # ? 1. Don't remove img links from annotations
                        # ? 2. The img html tag contains: alt, title and src as potential places that the PC could be found.
                        # ? 3. Find a way to recreate the img node into these three pieces and incoporate then into embedding
                        # for html_tag, xpath_content in node.items():
                        #     if text_annotation in xpath_content:
                        #         annotations_that_can_be_found.append(text_annotation)
                        #         break
                    if not found:
                        annotations_that_cannot_be_found.append(text_annotation)

                if len(annotations_that_cannot_be_found) > 0:
                    print(
                        f"{len(annotations_that_cannot_be_found)} {self.tag} cannot be found in {enum } \t: {annotations_that_cannot_be_found} - {url}"
                    )
                    print()

                all_annotations_left.append(annotations_that_can_be_found)
            else:
                all_annotations_left.append(None)

        final_amount_of_label = self.count_all_labels(df)
        print(f"Final amount of labels: {final_amount_of_label}")
        print(
            f"Number of labels lost because they couldn't be found in the page: {initial_amount_of_label - final_amount_of_label}"
        )

        df[f"{self.tag}-gt_text"] = all_annotations_left
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len)
        df = df[df[f"{self.tag}-gt_text_count"] > 0]
        print("done")
        return df

    def remove_folder(self, raw_data_folder):
        print("Remove folder...")
        self.raw_data_folder = raw_data_folder
        if os.path.exists(self.raw_data_folder):
            print(f"Overwriting this folder: \n{self.raw_data_folder}")
            try:
                shutil.rmtree(self.raw_data_folder)
                print(f"REMOVED: {self.raw_data_folder}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")

        groundtruth_folder_path = self.raw_data_folder
        groundtruth_folder_path.mkdir(parents=True, exist_ok=True)

    def add_gt_counts_and_sort(self, df):
        print("Add_gt_counts_and_sort...")
        df[f"{self.tag}-gt_text_counts"] = domain_df[f"{self.tag}-gt_text"].apply(len)
        return df.sort_values(f"{self.tag}-gt_text_counts", ascending=False)

    def add_page_id(self, df):
        df["page_id"] = [str(index).zfill(4) for index in range(len(df))]
        return df

    def save_ground_truth(self, df, root_folder, domain_name):
        """
        In domain folder save a single csv file with its pages annotations
        """
        print("Save_ground_truth...")

        folder_path = root_folder / "ground_truth"
        folder_path.mkdir(parents=True, exist_ok=True)

        page_annotations_df = df[["page_id", f"{self.tag}-gt_text_counts", f"{self.tag}-gt_text"]]
        page_annotations_df.to_csv(
            folder_path / f"{domain_name}-{self.tag}.csv", sep="\t", index=False
        )

    def save_htmls(self, df, root_folder, domain_name):
        """
        In domain folder save all html pages
        """

        def save_html(html, save_path):
            Html_file = open(save_path, "w")
            Html_file.write(html)
            Html_file.close()

        print("Save htmls...")
        folder_path = root_folder / "htmls" / domain_name
        folder_path.mkdir(parents=True, exist_ok=True)
        df.apply(lambda row: save_html(row["html"], folder_path / f"{row['page_id']}.htm"), axis=1)

    def save_domain_node_features(self, df, raw_data_folder, domain_name):
        folder_path = raw_data_folder / "prepared"
        folder_path.mkdir(parents=True, exist_ok=True)
        domain_nodes = []
        for page_nodes in df["nodes"]:
            domain_nodes.extend(page_nodes)
        domain_nodes_df = pd.DataFrame(domain_nodes, columns=["xpath", "text", "tag", "gt_texts"])
        domain_nodes_df.to_pickle(folder_path / f"{domain_name}.pkl")
        return domain_nodes_df

    def save_dedup(self, domain_nodes_df, raw_data_folder, domain_name):
        folder_path = raw_data_folder / "prepared_dedup"
        folder_path.mkdir(parents=True, exist_ok=True)

        domain_nodes_df.drop_duplicates("text").to_pickle(folder_path / f"{domain_name}.pkl")

    def add_classification_label(self, nodes, gt_texts):
        """
        Node: [(xpath, text), (...)]
        gt_texts: [gt_text1, gt_text2]
        Annotated_Node: [(xpath, text, tag, [gt_text1, gt_text2]), (...)]
        """

        nodes_annotated = []
        for xpath, node_text in nodes:
            gt_text_in_node = []
            for gt_text in gt_texts:
                if f" {gt_text.strip()} ".lower() in f" {node_text.strip()} ".lower():
                    gt_text_in_node.append(gt_text)

            if len(gt_text_in_node) == 0:
                new_node_text = (xpath, node_text, None, [])
            else:
                new_node_text = (
                    xpath,
                    node_text,
                    self.tag,
                    gt_text_in_node,
                )
            nodes_annotated.append(new_node_text)
        return nodes_annotated

    def add_classification_label_to_nodes(self, df):
        df["nodes"] = df.apply(
            lambda row: self.add_classification_label(row["nodes"], row[f"{self.tag}-gt_text"]),
            axis=1,
        )
        return df


if __name__ == "__main__":
    # wandb.login()
    # self.run = wandb.init(project="LanguageModel", resume=False, tags=["convert_data"])

    dataset_name = "develop"

    # wae_data_load_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}")
    wae_data_load_path = Path(f"/data/GIT/delete/")

    raw_data_folder = Path(f"/data/GIT/delete/{dataset_name}")

    prepare_data = PrepareData(tag="PAST_CLIENT")

    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    DOC_STRIDE = 128
    MAX_SEQ_LENGTH = 384
    featurizer = Featurizer(tokenizer=tokenizer, doc_stride=DOC_STRIDE, max_length=MAX_SEQ_LENGTH)

    prepare_data.remove_folder(raw_data_folder)

    df = prepare_data.load_data(wae_data_load_path, limit=-1)  # develop size = 75824

    # df["domain"] = df["domain"].apply(lambda domain: domain.replace("-", ""))
    # assert (
    #     len(df[df["domain"].apply(lambda domain: "(" in domain or ")" in domain)]) == 0
    # )  # ? Make sure domain names don't have parenthesis

    df = prepare_data.format_annotation(df)

    df_positives_negatives = prepare_data.create_postive_negative_data(df, 1)

    for domain_name, domain_df in df_positives_negatives.groupby("domain"):

        domain_df["html"] = domain_df.apply(
            lambda row: featurizer.insert_url_into_html(row["url"], row["html"]), axis=1
        )
        domain_df["nodes"] = domain_df["html"].apply(featurizer.get_nodes)

        domain_df = prepare_data.add_gt_counts_and_sort(domain_df)
        domain_df = prepare_data.add_page_id(domain_df)

        prepare_data.save_ground_truth(domain_df, raw_data_folder, domain_name)
        prepare_data.save_htmls(domain_df, raw_data_folder, domain_name)

        domain_df = prepare_data.add_classification_label_to_nodes(domain_df)

        domain_nodes_df = prepare_data.save_domain_node_features(
            domain_df, raw_data_folder, domain_name
        )
        prepare_data.save_dedup(domain_nodes_df, raw_data_folder, domain_name)

        domain_df["swde_features"] = domain_df["nodes"].apply(featurizer.get_swde_features)

    # self.run.save[""]()
    # self.run.finish()
