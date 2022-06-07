import torch
from urllib.parse import unquote, urlparse
import re
from lxml import etree
import lxml
from lxml.html.clean import Cleaner
import unicodedata
from markuplmft.fine_tuning.run_swde import constants
from markuplmft.data.tag_utils import tags_dict
from torch.utils.data import Dataset


class SwdeFeature(object):  # BatchEncoding PageClassifierFeature
    def __init__(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels,
        relative_first_tokens_node_index,
        absolute_node_index,
        url,
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
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.relative_first_tokens_node_index = relative_first_tokens_node_index
        self.absolute_node_index = absolute_node_index
        self.url = url


class SwdeDataset(Dataset):
    def __init__(
        self,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        relative_first_tokens_node_index,
        absolute_node_index,
        url,
        all_labels=None,
    ):

        self.tensors = [
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
        ]
        self.relative_first_tokens_node_index = relative_first_tokens_node_index
        self.absolute_node_index = absolute_node_index
        self.url = url

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

    def get_swde_features(self, nodes, url):
        """
        1. Tokenizer page
        2. Convert tokens into ids
        3. Split 384 tokens ids into features that go into the model
        4. Move features by a doc_stride of 128 
        """
        real_max_token_num = self.max_length - 2  # for cls and sep

        page_tokens_ids = []
        page_labels_seq = []
        # ? A list of indices pointing to the first tokens in each node of the all tokens list
        absolute_first_tokens_node_indices = []  

        # TODO: Move this to a function called tokenize_page_nodes
        # ? Create a tokenization of the page by converting the text of each node into token_ids and concatenate them.
        # ? For each node append the number of tokens to first_token_pos.
        for node in nodes:
            # xpath, node_text, tag, gt_text = node[0], node[1], node[2], node[3]
            node_text, gt_text = node[1], node[3]

            # ? Tokenize and convert tokens into ids
            node_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(node_text))
            # ? Concatenate token_ids
            node_tokens_ids = node_tokens_ids + [self.tokenizer.pad_token_id] # TODO: Check for a better token
            page_tokens_ids += node_tokens_ids 

            # ? Append the position of the first token of the node
            # ? We always use the first token to predict
            absolute_first_tokens_node_indices.append(len(page_labels_seq))  # ? E.g. len(all_labels_seq) = 71
            # ? E.g. first_tokens_node_index = [71, 95, 100, 104, 184, 192, 198, 212]

            if len(gt_text) > 0:
                annotation_type = "PAST_CLIENT"
            else:
                annotation_type = "none"

            page_labels_seq += [constants.ATTRIBUTES_PLUS_NONE.index(annotation_type)] * len(
                node_tokens_ids
            )
            # ? E. g. all_labels_seq = [1, 1, 1, 0, 1, 1, ..., 0, 1, 1]
            # ? The numbers in each token_ids indicates the label index in constants.ATTRIBUTES_PLUS_NONE
            # ? This means that all tokens_ids for the text in the xpath
            # ? will get labelled as something differently than -100 in case it is positive.

        number_of_nodes = len(nodes)
        features = []
        start_pos = 0
        page_over_flag = False

        node_index = 0

        # TODO (Aimore): Check if the nodes are being dropped somehow. It seems there are less nodes than it should be?
        while True:
            # ? This loop goes over page_tokens_ids in a stride manner.
            # ? Gets a subset of the page_tokens_ids and appends cls_token_id(0) at the beginning and cls_token_id(2) at the end.

            end_pos = start_pos + real_max_token_num
            
            splited_page_tokens_ids = (
                [self.tokenizer.cls_token_id]
                + page_tokens_ids[start_pos:end_pos]
                + [self.tokenizer.sep_token_id]
            )
            # ? tokenizer.cls_token_id = 0
            # ? tokenizer.sep_token_id = 2
            # ? The length of the subset is given by the real_max_token_num 382.
            # ? E.g. splited_token_ids [len(382)] = [42996, 4, 23687, 48159, 5457, 2931, ...]

            token_type_ids = [0] * self.max_length  # ? It is always a list of 0

            splited_labels_seq = [-100] + page_labels_seq[start_pos:end_pos] + [-100]
            #? E. g. splited_labels_seq = [-100, 1, 1, 1, 0, 1, 1, ..., 0, 1, 1, -100]
            #? This is to mask the CLS and SEP tokens

            relative_first_tokens_node_indices = []
            absolute_node_indices = []
            # involved_first_tokens_text = []

            # ? This while gets the first token node indices between the start and end window
            # ? This while doesn't run if relative_first_tokens_node_index[curr_first_token_index] is very high (above 382)
            # ? This while loops over the relative_first_tokens_node_index and breaks if relative_first_tokens_node_index is higher than the end_pos
            while (
                node_index < number_of_nodes
                and start_pos <= absolute_first_tokens_node_indices[node_index] < end_pos
            ):  

                # ? +1 because of the first token: [cls]
                relative_first_tokens_node_indices.append(
                    absolute_first_tokens_node_indices[node_index] - start_pos + 1
                )
                absolute_node_indices.append(node_index)
                node_index += 1

            if end_pos >= len(page_tokens_ids):
                # ? This will be the last time of this loop. When the page is over.
                page_over_flag = True
                # ? The first step is to get the features for the window, which means we need to pad in this feature with -100
                current_len = len(splited_page_tokens_ids)
                splited_page_tokens_ids += [self.tokenizer.pad_token_id] * (self.max_length - current_len)
                splited_labels_seq += [-100] * (self.max_length - current_len)
                attention_mask = [1] * current_len + [0] * (self.max_length - current_len)

            else:
                # ? no need to pad, the splited seq is exactly with the length `self.max_length`
                assert len(splited_page_tokens_ids) == self.max_length
                attention_mask = [1] * self.max_length

            # if len(relative_first_tokens_node_indices) == 0:
            #     print("EMPTY")

            features.append(
                SwdeFeature(
                    input_ids=splited_page_tokens_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=splited_labels_seq,
                    relative_first_tokens_node_index=relative_first_tokens_node_indices,
                    absolute_node_index=absolute_node_indices,
                    url=url
                )
            )
            start_pos += self.doc_stride

            if page_over_flag:
                break
        
        return features
        # ? features = [swde_feature_1, swde_feature_2, ...]

    def feature_to_dataset(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        relative_first_tokens_node_index = [f.relative_first_tokens_node_index for f in features]
        absolute_node_index = [f.absolute_node_index for f in features]
        url = [f.url for f in features]

        #! If there are labels then create with labels
        all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)

        dataset = SwdeDataset(all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        relative_first_tokens_node_index,
        absolute_node_index,
        url,
        all_labels)

        return dataset


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
