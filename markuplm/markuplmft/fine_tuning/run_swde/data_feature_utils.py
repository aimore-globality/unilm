import tqdm
from torch.utils.data import Dataset
from markuplmft.data.tag_utils import tags_dict
import pickle
import os
from markuplmft.fine_tuning.run_swde import constants


class SwdeFeature(object):
    def __init__(
        self,
        html_path,
        input_ids,
        token_type_ids,
        attention_mask,
        xpath_tags_seq,
        xpath_subs_seq,
        labels,
        involved_first_tokens_pos,
        involved_first_tokens_xpaths,
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
        xpath_tags_seq: RT
        xpath_subs_seq: RT
        labels: RT
        involved_first_tokens_pos: a list, indicate the positions of the first-tokens in this feature
        involved_first_tokens_xpaths: the xpaths of the first-tokens, used to build dict
        involved_first_tokens_types: the types of the first-tokens
        involved_first_tokens_text: the text of the first tokens

        Note that `involved_xxx` are not fixed-length array, so they shouldn't be sent into our model
        They are just used for evaluation
        """
        self.html_path = html_path
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.xpath_tags_seq = xpath_tags_seq
        self.xpath_subs_seq = xpath_subs_seq
        self.labels = labels
        self.involved_first_tokens_pos = involved_first_tokens_pos
        self.involved_first_tokens_xpaths = involved_first_tokens_xpaths
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
        all_xpath_tags_seq,
        all_xpath_subs_seq,
        all_labels=None,
    ):
        """
        print(type(all_input_ids))
        print(type(all_attention_mask))
        print(type(all_token_type_ids))
        print(type(all_xpath_tags_seq))
        print(type(all_xpath_subs_seq))
        print(type(all_labels))
        raise ValueError
        """
        self.tensors = [
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_xpath_tags_seq,
            all_xpath_subs_seq,
        ]

        if not all_labels is None:
            self.tensors.append(all_labels)

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


def process_xpath(xpath: str):
    if xpath.endswith("/tail"):
        xpath = xpath[:-5]

    xpath_tags_seq, xpath_subs_seq = [], []
    units = xpath.split("/")

    for unit in units:
        if not unit:
            continue
        if "[" not in unit:
            xpath_tags_seq.append(tags_dict.get(unit, 215))
            xpath_subs_seq.append(0)
        else:
            xx = unit.split("[")
            name = xx[0]
            id = int(xx[1][:-1])
            xpath_tags_seq.append(tags_dict.get(name, 215))
            xpath_subs_seq.append(min(id, 1000))

    assert len(xpath_subs_seq) == len(xpath_tags_seq)

    if len(xpath_tags_seq) > 50:
        xpath_tags_seq = xpath_tags_seq[-50:]
        xpath_subs_seq = xpath_subs_seq[-50:]

    xpath_tags_seq = xpath_tags_seq + [216] * (50 - len(xpath_tags_seq))
    xpath_subs_seq = xpath_subs_seq + [1001] * (50 - len(xpath_subs_seq))

    return xpath_tags_seq, xpath_subs_seq


def get_swde_features(
    root_dir,
    website,
    tokenizer,
    doc_stride,
    max_length,
):
    """
    This function creates a list of features that goes into the model.
    The data already comes divided by nodes.
    Here, most of the fixed nodes are filtered out.
    From the remaining nodes, features are created.
    Each feature has 384 tokens. The sum of the
    Each feature represents a

    """
    real_max_token_num = max_length - 2  # for cls and sep
    padded_xpath_tags_seq = [216] * 50
    padded_xpath_subs_seq = [1001] * 50

    filename = os.path.join(root_dir, website)
    with open(filename, "rb") as f:
        raw_data = pickle.load(f)

    features = []

    # This loops goes over all the pages in a website
    # for page_id in tqdm.tqdm(raw_data, desc=f"Processing {website} features ...", leave=False):
    for page_id in raw_data:
        html_path = f"{website}-{page_id}.htm"

        all_token_ids_seq = []
        all_xpath_tags_seq = []
        all_xpath_subs_seq = []
        token_to_ori_map_seq = []
        all_labels_seq = []

        first_token_pos = []
        first_token_xpaths = []
        first_token_type = []
        first_token_text = []
        first_token_gt_text = []
        first_token_node_attribute = []
        first_token_node_tag = []

        needed_docstrings_id_list = sorted(list(range(len(raw_data[page_id]))))
        # E. g. needed_docstrings_id_list = [3, 5, 518, 522, 526, 659, 663, 667, 672, 675, 561, 570, 574, 578, 588, 593, 594, 595, 596, 597, 598, 509]
        # This is going to be a set of nodes in which the model will use. In case there is not fixed node, the model will use all nodes.

        # This for loop goes over the selected nodes and append the tokens and xpaths from each node
        for i, needed_id in enumerate(needed_docstrings_id_list):
            node_tag = raw_data[page_id][needed_id][5]
            node_attribute = raw_data[page_id][needed_id][4]
            gt_text = raw_data[page_id][needed_id][3]

            text = raw_data[page_id][needed_id][0]
            # ? E.g. text [str] = 'HITT FUTURES'
            xpath = raw_data[page_id][needed_id][1]
            # ? E.g. xpath [str] = '/html/body/div/div/div[2]/div[1]/div[2]/div/div/ul/li[3]/a'
            type = raw_data[page_id][needed_id][2]
            # ? E.g. type [str] = 'fixed-node'
            token_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text)
            )  # Here is where the text get converted into tokens ids
            # ? E.g. tokenizer.tokenize breaks the text into smaller pieces, and the tokenizer.convert_tokens_to_ids applied a map key-value
            # There seems to be no cutdown of size/text.
            # ? E.g. token_ids = [725, 23728, 274, 6972, 28714]

            xpath_tags_seq, xpath_subs_seq = process_xpath(xpath)
            # ? E.g. xpath_tags_seq [len(50)] = [109, 25, 50, 50, 50, 50, 50, 50, 50, 207, 120, 0, 216, 216, 216, ...]
            # ? E.g. xpath_subs_seq [len(50)] = [0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 3, 0, 1001, 1001, 1001, 1001, ...]

            all_token_ids_seq += token_ids
            # all_token_ids_seq is the sequence of tokens from all the selected nodes (fixed, variable and fround truth)
            # E. g. all_token_ids_seq = [42996, 4, 23687, 48159, 5457, 2931, ... 43163, 1297, 22, 29766, 12, 698, 34414]
            all_xpath_tags_seq += [xpath_tags_seq] * len(token_ids)
            # E. g. all_xpath_tags_seq len(95)[len(50)] max 1001 = [[0, 0, 2, 1001, 1001, 1001, 1001,...], [...], [...], ...]
            all_xpath_subs_seq += [xpath_subs_seq] * len(token_ids)
            # E. g. all_xpath_subs_seq len(95)[len(50)] max 216 = [[109, 104, 169, 216, 216, 216, 216, ,...], [...], [...], ...]

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
        assert len(all_token_ids_seq) == len(all_xpath_subs_seq)
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

            token_type_ids = [0] * max_length  # that is always this

            end_pos = start_pos + real_max_token_num
            # add start_pos ~ end_pos as a feature
            splited_token_ids_seq = (
                [tokenizer.cls_token_id]
                + all_token_ids_seq[start_pos:end_pos]
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
            splited_xpath_subs_seq = (
                [padded_xpath_subs_seq]
                + all_xpath_subs_seq[start_pos:end_pos]
                + [padded_xpath_subs_seq]
            )
            splited_labels_seq = [-100] + all_labels_seq[start_pos:end_pos] + [-100]

            # ? locate first-tokens in them
            involved_first_tokens_pos = []
            involved_first_tokens_xpaths = []
            involved_first_tokens_types = []
            involved_first_tokens_text = []
            involved_first_tokens_gt_text = []
            involved_first_tokens_node_attribute = []
            involved_first_tokens_node_tag = []

            while (
                curr_first_token_index < len(first_token_pos)
                and end_pos > first_token_pos[curr_first_token_index] >= start_pos
            ):  # ? This while doesn't run if first_token_pos[curr_first_token_index] is very high (above 382)
                # ? This while loops over the first_token_pos and breaks if first_token_pos is higher than the end_pos
                involved_first_tokens_pos.append(
                    first_token_pos[curr_first_token_index] - start_pos + 1
                )  # ? +1 for [cls]
                involved_first_tokens_xpaths.append(first_token_xpaths[curr_first_token_index])
                involved_first_tokens_types.append(first_token_type[curr_first_token_index])
                involved_first_tokens_text.append(first_token_text[curr_first_token_index])
                involved_first_tokens_gt_text.append(first_token_gt_text[curr_first_token_index])
                involved_first_tokens_node_attribute.append(
                    first_token_node_attribute[curr_first_token_index]
                )
                involved_first_tokens_node_tag.append(first_token_node_tag[curr_first_token_index])
                curr_first_token_index += 1

            if end_pos >= len(all_token_ids_seq):
                # ? This will be the last time of this loop.
                flag = True
                # ? which means we need to pad in this feature
                current_len = len(splited_token_ids_seq)
                splited_token_ids_seq += [tokenizer.pad_token_id] * (max_length - current_len)
                splited_xpath_tags_seq += [padded_xpath_tags_seq] * (max_length - current_len)
                splited_xpath_subs_seq += [padded_xpath_subs_seq] * (max_length - current_len)
                splited_labels_seq += [-100] * (max_length - current_len)
                attention_mask = [1] * current_len + [0] * (max_length - current_len)

            else:
                # ? no need to pad, the splited seq is exactly with the length `max_length`
                assert len(splited_token_ids_seq) == max_length
                attention_mask = [1] * max_length

            features.append(
                SwdeFeature(  # TODO (Aimore): If you put a breakpoint here you will see that many features are being created with empty text  -verify why
                    html_path=html_path,
                    input_ids=splited_token_ids_seq,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    xpath_tags_seq=splited_xpath_tags_seq,
                    xpath_subs_seq=splited_xpath_subs_seq,
                    labels=splited_labels_seq,
                    involved_first_tokens_pos=involved_first_tokens_pos,
                    involved_first_tokens_xpaths=involved_first_tokens_xpaths,
                    involved_first_tokens_types=involved_first_tokens_types,
                    involved_first_tokens_text=involved_first_tokens_text,
                    involved_first_tokens_gt_text=involved_first_tokens_gt_text,
                    involved_first_tokens_node_attribute=involved_first_tokens_node_attribute,
                    involved_first_tokens_node_tag=involved_first_tokens_node_tag,
                )
            )
            # TODO (aimore): It seems that the stride is implemented wrong here.
            # ?  Instead of doing start_pos += doc_stride.
            # ?  In this way the stride will be 254, instead of 128.
            # ?  So the first time is from 0 to 386, Then 508

            start_pos = end_pos - doc_stride
                        
            if flag:
                break

    return features
    # ? features = [swde_feature_1, swde_feature_2, ...]
