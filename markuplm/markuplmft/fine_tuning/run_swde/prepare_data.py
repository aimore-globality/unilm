# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extracting XPaths of the values of all fields for SWDE dataset."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import os
import pickle
import random
import re
import sys
import unicodedata

from absl import app
from absl import flags
import lxml
from lxml import etree
from lxml.html.clean import Cleaner
from tqdm import tqdm

import multiprocessing as mp
from pathlib import Path
import glob

FLAGS = flags.FLAGS
random.seed(42)

flags.DEFINE_string(
    "input_groundtruth_path",
    "",
    "The root path to parent folder of all ground truth files.",
)

flags.DEFINE_string(
    "input_pickle_path",
    "",
    "The root path to pickle file of swde html content.",
)

flags.DEFINE_string(
    "output_data_path",
    "",
    "The path of the output file containing both the input sequences and "
    "output sequences of the sequence tagging version of swde dataset.",
)

def load_html_and_groundtruth(website_to_load):
    #! All this function will be replaced
    """Loads and returns the html string and ground truth data as a dictionary."""
    all_data_dict = collections.defaultdict(dict)

    gt_path = FLAGS.input_groundtruth_path

    print("Reading the SWDE dataset pickle...", file=sys.stderr)
    with open(FLAGS.input_pickle_path, "rb") as load_file:
        swde_html_data = pickle.load(load_file)
    # swde_html_data =[ {"website":'book-amazon(2000)', "path:'book/book-amazon(2000)/0000.htm', "html_str":xx}, ...]

    # TODO (aimore): Here when this dataset is transformed into pandas this will change and it will be easier
    file_expression = os.path.join(gt_path, website_to_load) + "**"
    files_with_exp = glob.glob(file_expression)
    for file in files_with_exp:
        # For example, a groundtruth file name can be "yahoo-PAST_CLIENT.txt".
        website, field = file.replace(".txt", "").split("-")

        with open(os.path.join(gt_path, file), "r") as load_file:
            lines = load_file.readlines()
            for line in lines[2:]:
                # Each line should contains more than 3 elements splitted by \t
                # which are: page_id, number of values, value1, value2, etc.
                gt_items = line.strip().split("\t")
                page_id = gt_items[0]

                gt_texts = [
                    gt_text
                    for gt_text in gt_items[2:]
                    if len(gt_text) > 0
                ]
                all_data_dict[page_id][f"field-{field}"] = dict(values=gt_texts)

                website_data = swde_html_data[website_to_load]

                all_data_dict[page_id]["path"] = website_data[page_id]["path"]
                all_data_dict[page_id]["html_str"] = website_data[page_id]["html_str"]

        # {"0000":
        #   {"field-PAST_CLIENT":
        #       {"values":["Tishman Speyer","Tishman Speyer2"]},
        #   }
        # }
    return all_data_dict


def get_field_xpaths(
    all_data_dict,
):
    """Gets xpaths data for each page in the data dictionary.

    Args:
      all_data_dict: the dictionary saving both the html content and the truth.
    """
    # Saving the xpath info of the whole website,
    #  - Key is a xpath.
    #  - Value is a set of text appeared before inside the node.
    overall_xpath_dict = collections.defaultdict(set)

    # current_xpath_data = dict() # I added this condition in case the page doesn't contain any positive annotation
    # current_page_nodes_in_order = [] # I added this condition in case the page doesn't contain any positive annotation

    #  Update page data with groundtruth xpaths and the overall xpath-value dict.
    for page_id in tqdm(all_data_dict):
        # We add dom-tree attributes for the first n_pages
        html = all_data_dict[page_id]["html_str"]

        # dom_tree = get_dom_tree(html, website=website_to_process)
        dom_tree = etree.ElementTree(lxml.html.fromstring(html))

        all_data_dict[page_id]["dom_tree"] = dom_tree

        # Match values of each field for the current page.
        fields = [
            keys
            for keys in all_data_dict[page_id]
            if "field" in keys
        ]  # all_data_dict[page_id] = ['field-PAST_CLIENT', 'path', 'html_str', 'dom_tree']
        for field in fields:
            # Saving the xpaths of the values for each field.
            all_data_dict[page_id][field]["groundtruth_xpaths"] = set()
            all_data_dict[page_id][field]["is_truth_value_list"] = set()

            gt_values = all_data_dict[page_id][field]["values"]
            #? Clean the groundtruth gt_values
            clean_gt_values = gt_values
            # clean_gt_values = []
            # for gt_value in gt_values:
            #     # Some gt_values contains HTML tags and special strings like "&nbsp;"
            #     # So we need to escape the HTML by parsing and then extract the inner text.
            #     gt_value = lxml.html.fromstring(gt_value)
            #     gt_value = " ".join(etree.XPath("//text()")(gt_value))
            #     gt_value = clean_spaces(gt_value)
            #     gt_value = clean_format_str(gt_value)
            #     gt_value = gt_value.strip()
            #     clean_gt_values.append(gt_value)

            matched_xpaths = []  # The resulting list of xpaths to be returned.
            current_xpath_data = dict()  # The resulting dictionary to save all page data.

            gt_text_in_nodes = dict()  # A list of the gt_text in each xpath node

            current_page_nodes_in_order = []
            is_truth_value_list = []

            # ? Iterate all the nodes in the given DOMTree.
            for node in dom_tree.iter():
                # The value can only be matched in the text of the node or the tail.
                node_text_dict = {
                    "node_text": node.text,
                    "node_tail_text": node.tail,
                }  # ?The only nodes that are considered here are the node.text and node.tail

                for text_part_flag, node_text in node_text_dict.items():
                    if node_text:
                        if (
                            node.tag != "script"
                            and "javascript" not in node.attrib.get("type", "")
                            and min_node_text_size <= len(node_text.strip()) < max_node_text_size
                        ):  #! Remove java/script and min_node_text # TODO (Aimore): Make this comparisons more explicity and descriptive
                            # """Matches the ground truth value with a specific node in the domtree.

                            # In the function, the current_xpath_data, overall_xpath_dict, matched_xpaths will be updated.

                            # Args:
                            # is_truth_value_list: [], indicate which node is the truth-value
                            # current_page_nodes_in_order: [(text, xpath)] seq
                            # node: the node on the domtree that we are going to match.
                            # node_text: the text inside this node.
                            # current_xpath_data: the dictionary of the xpaths of the current domtree.
                            # overall_xpath_dict: the dictionary of the xpaths of the current website.
                            # # text_part_flag: to match the "text" or the "tail" part of the node.
                            # groundtruth_value: the value of our interest to match.
                            # matched_xpaths: the existing matched xpaths list for this value on domtree.
                            # website: the website where the value is from.
                            # field: the field where the value is from.
                            # dom_tree: the current domtree object, used for getting paths.
                            # """
                            # Dealing with the cases with multiple <br>s in the node text,
                            # where we need to split and create new tags of matched_xpaths.
                            # For example, "<div><span>asd<br/>qwe</span></div>"

                            node_attribute = node.attrib.get("type", "")
                            node_tag = node.tag
                            node_text_split = node_text.split("--BRRB--")
                            len_brs = len(node_text_split)  # The number of the <br>s.
                            for index, etext in enumerate(node_text_split):

                                if text_part_flag == "node_text":
                                    xpath = dom_tree.getpath(node)

                                elif text_part_flag == "node_tail_text":
                                    xpath = dom_tree.getpath(node) + "/tail"

                                if len_brs >= 2:
                                    xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]

                                # clean_etext = clean_spaces(etext)
                                clean_etext = etext

                                # ? Update the dictionary.
                                current_xpath_data[xpath] = clean_etext
                                overall_xpath_dict[xpath].add(clean_etext)
                                current_page_nodes_in_order.append(
                                    (clean_etext, xpath, node_attribute, node_tag)
                                )

                                # ? Clean the groundtruth and the node text. Check if the groundtruth is in the node text.
                                # clean_etext = clean_format_str(clean_etext)

                                # ? Create node ground truth by checking if the the gt_text is in the clean node_text
                                gt_text_in_node = []
                                for gt_value in clean_gt_values:
                                    if f" {gt_value.strip()} ".lower() in f" {clean_etext.strip()} ".lower():
                                        gt_text_in_node.append(gt_value)
                                        matched_xpaths.append(xpath)
                                        is_truth_value_list.append(
                                            len(current_page_nodes_in_order) - 1
                                        )
                                        # break #! I am not sure why Iadded this break, I'm commenting it because I think all gt_values should be added in a node

                                if len(matched_xpaths) == 0:
                                    gt_text_in_nodes[xpath] = []
                                else:
                                    gt_text_in_nodes[xpath] = gt_text_in_node

            # ? Update the page-level xpath information.
            all_data_dict[page_id][field]["groundtruth_xpaths"].update(matched_xpaths)
            all_data_dict[page_id][field]["is_truth_value_list"].update(is_truth_value_list)

            all_data_dict[page_id][field]["gt_text_in_nodes"] = gt_text_in_nodes

            #? now for each all_data_dict[page_id]
            #? an example
            #? all_data_dict[page_id]["field-PAST_CLIENT"] =
            #? {
            #?   'values': ['Dave Kemper', 'Patrick Sebranek', 'Verne Meyer'],
            #?   'groundtruth_xpaths':
            #?       {'/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[3]',
            #?        '/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[2]',
            #?        '/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[1]',
            #?        '/html/body/div[2]/div[2]/div[3]/div[3]/p/a'}
            #? }

        all_data_dict[page_id]["xpath_data"] = current_xpath_data  #? {xpath1: text1, xpath2: text2}
        all_data_dict[page_id]["doc_strings"] = current_page_nodes_in_order  #? [(text, xpath)*N]
        #? all_data_dict[page_id]["reversed_doc_strings_ids"] = {v[0]: i for i, v in enumerate(current_page_nodes_in_order)}

    #? all_data_dict[page_id]["doc_strings"] is the basis of our transformers-based method!!!

    variable_nodes = set(overall_xpath_dict.keys())

    assert len(variable_nodes) > 0
    print(
        f"Website: {website_to_process} | Across all pages:\n \tvariable_nodes: {len(variable_nodes)}"
    )

    all_data_dict["variable_nodes"] = list(variable_nodes)

    return all_data_dict

def generate_nodes_seq_and_write_to_file(website):
    """Extracts all the xpaths and labels the nodes for all the pages."""

    all_data_dict = load_html_and_groundtruth(website) #! All this function will be replaced
    """
    all_data_dict = {'0000': {
    field-PAST_CLIENT: {'values': ['we work', 'sse', 'Oman Investment Corporation']}, 
    'path':'addleshawgoddard.com(105)/0000.htm', 
    'html_str':'html' }, 
    '0001': {...} }}
    """

    all_data_dict = get_field_xpaths(
        all_data_dict=all_data_dict,
    )

    """
    all_data_dict = {
        '0000': {
            field-PAST_CLIENT: {
                'values': ['SA', 'Luye Pharma Group Ltd.', 'Vipshop (US) Inc', 'Delta Air Lines', 'New York University', 'Harmon Store', 'Rutgers University', 'Amneal Pharmaceuticals, LLC', 'Kashiv Pharma, LLC', ...], 
                'groundtruth_xpaths': {'/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[11]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[12]/h2', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[21]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[19]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[3]/h2', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[20]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[10]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[3]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[18]/p[1]', ...}, 
                'is_truth_value_list': {384, 256, 128, 649, 533, 410, 412, 669, 555, ...}
                'gt_text_in_nodes': {xpath1:[], xpath2:[gt_text1, gt_text2]}
                }, 
            'path': 'lernerdavid.com(41)/0000.htm', 
            'html_str': 'html' }, 
            'dom_tree': dom_tree
            'xpath_data': {'/html/head': '', '/html/head/script[1]/tail': '', '/html/head/script[2]': 'window.jQuery || document.write(\'<script src="/Darwin/script/jquery/jquery-1.11.2.min.js"><\\/script>\')', '/html/head/script[2]/tail': '', '/html/head/script[3]/tail': '', '/html/head/script[4]': 'window.jQuery.ui || document.write(\'<script src="/Darwin/script/jquery/jquery-ui-1.12.1.min.js"><\\/script>\')', '/html/head/script[4]/tail': '', '/html/head/title': 'Successes', '/html/body': '', '/html/body/noscript[1]/tail': '', '/html/body/noscript[1]/div': 'Javascript must be enabled for the correct page display', '/html/body/script[1]': "//<![CDATA[var theForm = document.forms['aspnetForm'];if (!theForm) { theForm = document.aspnetForm;}function __doPostBack(eventTarget, eventArgument) { if (!theForm.onsubmit || (theForm.onsubmit() != false)) { theForm.__EVENTTARGET.value = eventTarget; theForm.__EVENTARGUMENT.value = eventArgument; theForm.submit(); }}//]]>", '/html/body/script[1]/tail': '', '/html/body/div[2]': '', ...}
            'doc_strings': [('', '/html/head'), ('', '/html/head/script[1]/tail'), ('window.jQuery || document.write(\'<script src="/Darwin/script/jquery/jquery-1.11.2.min.js"><\\/script>\')', '/html/head/script[2]'), ('', '/html/head/script[2]/tail'), ('', '/html/head/script[3]/tail'), ('window.jQuery.ui || document.write(\'<script src="/Darwin/script/jquery/jquery-ui-1.12.1.min.js"><\\/script>\')', '/html/head/script[4]'), ('', '/html/head/script[4]/tail'), ('Successes', '/html/head/title'), ('', '/html/body'), ('', '/html/body/noscript[1]/tail'), ('Javascript must be enabled for the correct page display', '/html/body/noscript[1]/div'), ("//<![CDATA[var theForm = document.forms['aspnetForm'];if (!theForm) { theForm = document.aspnetForm;}function __doPostBack(eventTarget, eventArgument) { if (!theForm.onsubmit || (theForm.onsubmit() != false)) { theForm.__EVENTTARGET.value = eventTarget; theForm.__EVENTARGUMENT.value = eventArgument; theForm.submit(); }}//]]>", '/html/body/script[1]'), ('', '/html/body/script[1]/tail'), ('', '/html/body/div[2]'), ...]
        '0001': {
            ...
            } 
    },
    """

    cleaned_features_for_this_website = {}

    page_ids = [
        keys
        for keys in all_data_dict
        if "nodes" not in keys
    ]
    for page_id in page_ids:
        page_data = all_data_dict[page_id]
        gt_text_dict = page_data["field-PAST_CLIENT"]["gt_text_in_nodes"]

        assert "xpath_data" in page_data

        doc_strings = page_data["doc_strings"]
        # E.g. doc_strings = [('1730 Pennsylvania Avenue NW | HITT', '/html/head/title'), ...]
        new_doc_strings = []
        # The difference between doc_strings and new_doc_strings is that new_doc_strings contains the label
        field_info = {}
        # TODO(Aimore): Probably change the name of this variable (field_info) to something more meaningful, such as: gt_node_ids
        for field in [
            x
            for x in list(page_data.keys())
            if "field" in x
        ]:
            for doc_string_id in page_data[field]["is_truth_value_list"]:
                field_info[doc_string_id] = field.split("-")[1]

        for id, doc_string in enumerate(doc_strings):
            text, xpath, node_attribute, node_tag = doc_string
            gt_text = gt_text_dict.get(xpath)

            gt_field = field_info.get(id, "none")  # Choose between none or gt label (PAST_CLIENT)
            new_doc_strings.append((text, xpath, gt_field, gt_text, node_attribute, node_tag)) #? Here is what is inside the prepared data

        cleaned_features_for_this_website[page_id] = new_doc_strings

    output_file_path = os.path.join(FLAGS.output_data_path, f"{website}.pickle")
    print(
        f"Writing the processed and {len(cleaned_features_for_this_website)} pages of {website} into {output_file_path}"
    )
    nodes_per_page = {
        x[0]: len(x[1])
        for x in cleaned_features_for_this_website.items()
    }
    print(f"# Nodes per page: {nodes_per_page}\n")
    with open(output_file_path, "wb") as f:  #!UNCOMMENT
        pickle.dump(cleaned_features_for_this_website, f)


def main(_):
    if not os.path.exists(FLAGS.output_data_path):
        os.makedirs(FLAGS.output_data_path)

    swde_path = FLAGS.input_groundtruth_path.split("groundtruth")[0]
    p = Path(swde_path) / "WAE"
    websites = sorted([x.parts[-1].split("-")[-1].split("(")[0] for x in list(p.iterdir())])

    websites = websites[:]  #! Limit the amount of websites
    websites = [
        x
        for x in websites
        if "ciphr.com" not in x
    ]  #! Remove this website for now just because it is taking too long (+20min.)

    print(f"Prepare Data: Websites {len(websites)} -\n {websites}")

    global min_node_text_size
    min_node_text_size = 2
    global max_node_text_size
    max_node_text_size = 10_000

    debug = False
    if debug:
        for website in websites[:]:
            if website in ["canelamedia.com"]:
                # if 'direct.com' in website: # TODO (AIMORE): Debug this page and understand why it doesn't have variable nodes
                generate_nodes_seq_and_write_to_file(website)
    else:
        num_cores = mp.cpu_count()
        with mp.Pool(num_cores) as pool, tqdm(
            total=len(websites), desc="Processing swde-data"
        ) as t:
            # for res in pool.imap_unordered(generate_nodes_seq_and_write_to_file, websites):
            for res in pool.imap(generate_nodes_seq_and_write_to_file, websites):
                t.update()


if __name__ == "__main__":
    app.run(main)
