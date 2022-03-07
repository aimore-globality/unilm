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

flags.DEFINE_integer(
    "n_pages",
    2000,
    "The maximum number of pages to read.",
)

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


def match_value_node(
    node,
    node_text,
    current_xpath_data,
    overall_xpath_dict,
    text_part_flag,
    groundtruth_value,
    matched_xpaths,
    dom_tree,
    current_page_nodes_in_order,
    is_truth_value_list,
):
    """Matches the ground truth value with a specific node in the domtree.

    In the function, the current_xpath_data, overall_xpath_dict, matched_xpaths
    will be updated.

    Args:
      is_truth_value_list: [], indicate which node is the truth-value
      current_page_nodes_in_order: [(text, xpath)] seq
      node: the node on the domtree that we are going to match.
      node_text: the text inside this node.
      current_xpath_data: the dictionary of the xpaths of the current domtree.
      overall_xpath_dict: the dictionary of the xpaths of the current website.
      text_part_flag: to match the "text" or the "tail" part of the node.
      groundtruth_value: the value of our interest to match.
      matched_xpaths: the existing matched xpaths list for this value on domtree.
      website: the website where the value is from.
      field: the field where the value is from.
      dom_tree: the current domtree object, used for getting paths.
    """
    assert text_part_flag in ["node_text", "node_tail_text"]
    # Dealing with the cases with multiple <br>s in the node text,
    # where we need to split and create new tags of matched_xpaths.
    # For example, "<div><span>asd<br/>qwe</span></div>"
    len_brs = len(node_text.split("--BRRB--"))  # The number of the <br>s.
    for index, etext in enumerate(node_text.split("--BRRB--")):
        if text_part_flag == "node_text":
            xpath = dom_tree.getpath(node)
        elif text_part_flag == "node_tail_text":
            # TODO (aimore): I am not sure why they make the distinction of the
            #  xpath being the text or tail. That increases complexity of xpath and
            #  might not help during generalization.
            xpath = dom_tree.getpath(node) + "/tail"
        if len_brs >= 2:
            xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]
        clean_etext = clean_spaces(etext)

        # Update the dictionary.
        current_xpath_data[xpath] = clean_etext
        overall_xpath_dict[xpath].add(clean_etext)
        current_page_nodes_in_order.append((clean_etext, xpath))

        # Clean the groundtruth and the node text. Check if the groundtruth is in the node text.
        groundtruth_value = clean_format_str(groundtruth_value)
        clean_etext = clean_format_str(clean_etext)

        if groundtruth_value.strip() in clean_etext.strip():
            matched_xpaths.append(xpath)
            is_truth_value_list.append(len(current_page_nodes_in_order) - 1)

        # 这里我们更新三样东西 (Here we update three things)
        # 如果当前节点与truth_value一致 (If the current node is consistent with truth_value)，
        # 则将当前xpath加入matched_xpaths (Add the current xpath to matched_xpaths)
        # 此外，还需要 current_xpath_data[xpath] = clean_etext,即记录当前页面 该xpath对应的文字
        # (In addition, current_xpath_data[xpath] = clean_etext is also required,
        # that is, the text corresponding to the xpath of the current page is recorded)
        # 以及 overall_xpath_dict[xpath].add(clean_etext)，即记录当前网址上该xpath对应的文字，以add加入集合
        # (And overall_xpath_dict[xpath].add(clean_etext), that is,
        # record the text corresponding to the xpath on the current URL, and add it to the set)
    return current_xpath_data, overall_xpath_dict, current_page_nodes_in_order, matched_xpaths, is_truth_value_list


def get_value_xpaths(
    dom_tree,
    truth_value,
    overall_xpath_dict,
):
    """Gets a list of xpaths that contain a text truth_value in DOMTree objects.

    Args:
      dom_tree: the DOMTree object of a specific HTML page.
      truth_value: a certain groundtruth value.
      overall_xpath_dict: a dict maintaining all xpaths data of a website.

    Returns:
      xpaths: a list of xpaths containing the truth_value exactly as inner texts.
      current_xpath_data: the xpaths and corresponding values in this DOMTree.
    """

    matched_xpaths = []  # The resulting list of xpaths to be returned.
    current_xpath_data = dict()  # The resulting dictionary to save all page data.

    current_page_nodes_in_order = []
    is_truth_value_list = []

    # Some values contains HTML tags and special strings like "&nbsp;"
    # So we need to escape the HTML by parsing and then extract the inner text.
    value_dom = lxml.html.fromstring(truth_value)
    value = " ".join(etree.XPath("//text()")(value_dom))
    value = clean_spaces(value)

    # Iterate all the nodes in the given DOMTree.
    for node in dom_tree.iter():
        # The value can only be matched in the text of the node or the tail.
        if node.text:
            current_xpath_data, overall_xpath_dict, current_page_nodes_in_order, matched_xpaths, is_truth_value_list = match_value_node(
                node=node,
                node_text=node.text,
                current_xpath_data=current_xpath_data,
                overall_xpath_dict=overall_xpath_dict,
                text_part_flag="node_text",
                groundtruth_value=value,
                matched_xpaths=matched_xpaths,
                dom_tree=dom_tree,
                current_page_nodes_in_order=current_page_nodes_in_order,
                is_truth_value_list=is_truth_value_list,
            )
        if node.tail:
            current_xpath_data, overall_xpath_dict, current_page_nodes_in_order, matched_xpaths, is_truth_value_list = match_value_node(
                node=node,
                node_text=node.tail,
                current_xpath_data=current_xpath_data,
                overall_xpath_dict=overall_xpath_dict,
                text_part_flag="node_tail_text",
                groundtruth_value=value,
                matched_xpaths=matched_xpaths,
                dom_tree=dom_tree,
                current_page_nodes_in_order=current_page_nodes_in_order,
                is_truth_value_list=is_truth_value_list,
            )

    return current_xpath_data, overall_xpath_dict, current_page_nodes_in_order, matched_xpaths, is_truth_value_list


def get_dom_tree(html, website):
    """Parses a HTML string to a DOMTree.

    We preprocess the html string and use lxml lib to get a tree structure object.

    Args:
      html: the string of the HTML document.
      website: the website name for dealing with special cases.

    Returns:
      A parsed DOMTree object using lxml library.
    """
    cleaner = Cleaner()
    # cleaner.javascript = True
    cleaner.javascript = False
    # cleaner.scripts = True
    cleaner.scripts = False

    cleaner.style = True
    cleaner.page_structure = False
    html = html.replace("\0", "")  # Delete NULL bytes.
    # Replace the <br> tags with a special token for post-processing the xpaths.
    html = html.replace("<br>", "--BRRB--")
    html = html.replace("<br/>", "--BRRB--")
    html = html.replace("<br />", "--BRRB--")
    html = html.replace("<BR>", "--BRRB--")
    html = html.replace("<BR/>", "--BRRB--")
    html = html.replace("<BR />", "--BRRB--")

    # A special case in this website, where the values are inside the comments.
    if website == "careerbuilder":
        html = html.replace("<!--<tr>", "<tr>")
        html = html.replace("<!-- <tr>", "<tr>")
        html = html.replace("<!--  <tr>", "<tr>")
        html = html.replace("<!--   <tr>", "<tr>")
        html = html.replace("</tr>-->", "</tr>")

    html = clean_format_str(html)
    # TODO(Aimore): Deal with XML cases. If there are problems here with XLM, is because it can only treat HTMLpages
    x = lxml.html.fromstring(html)
    etree_root = cleaner.clean_html(x)
    dom_tree = etree.ElementTree(etree_root)
    return dom_tree


def load_html_and_groundtruth(website_to_load):
    """Loads and returns the html string and ground truth data as a dictionary."""
    all_data_dict = collections.defaultdict(dict)

    gt_path = FLAGS.input_groundtruth_path

    print("Reading the SWDE dataset pickle...", file=sys.stderr)
    with open(FLAGS.input_pickle_path, "rb") as load_file:
        swde_html_data = pickle.load(load_file)
    # swde_html_data =[ {"website":'book-amazon(2000)', "path:'book/book-amazon(2000)/0000.htm', "html_str":xx}, ...]

    # TODO (aimore): Here when this dataset is transformed into pandas this will change and it will be easier
    file_expression = os.path.join(gt_path, website_to_load) + '**'
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
    website_to_process,
):
    """Gets xpaths data for each page in the data dictionary.

    Args:
      all_data_dict: the dictionary saving both the html content and the truth.
      website_to_process: the website that we are working on.
    """
    # Saving the xpath info of the whole website,
    #  - Key is a xpath.
    #  - Value is a set of text appeared before inside the node.
    overall_xpath_dict = collections.defaultdict(set)

    # current_xpath_data = dict() # I added this condition in case the page doesn't contain any positive annotation
    # current_page_nodes_in_order = [] # I added this condition in case the page doesn't contain any positive annotation

    #  Update page data with groundtruth xpaths and the overall xpath-value dict.
    for page_id in tqdm(all_data_dict, desc=f"Processing: {website_to_process}", ):
        # We add dom-tree attributes for the first n_pages
        html = all_data_dict[page_id]["html_str"]

        dom_tree = get_dom_tree(html, website=website_to_process)
        all_data_dict[page_id]["dom_tree"] = dom_tree

        # Match values of each field for the current page.
        fields = [keys for keys in all_data_dict[page_id] if 'field' in keys] # all_data_dict[page_id] = ['field-PAST_CLIENT', 'path', 'html_str', 'dom_tree']
        for field in fields:            
            # Saving the xpaths of the values for each field.
            all_data_dict[page_id][field]["groundtruth_xpaths"] = set()
            all_data_dict[page_id][field]["is_truth_value_list"] = set()

            gt_values = all_data_dict[page_id][field]["values"]



            # Clean the groundtruth gt_values
            clean_gt_values = []
            for gt_value in gt_values: 
                # Some gt_values contains HTML tags and special strings like "&nbsp;"
                # So we need to escape the HTML by parsing and then extract the inner text.
                gt_value = lxml.html.fromstring(gt_value)
                gt_value = " ".join(etree.XPath("//text()")(gt_value))
                gt_value = clean_spaces(gt_value)
                gt_value = clean_format_str(gt_value)
                clean_gt_values.append(gt_value)



            matched_xpaths = []  # The resulting list of xpaths to be returned.
            current_xpath_data = dict()  # The resulting dictionary to save all page data.

            current_page_nodes_in_order = []
            is_truth_value_list = []


            # Iterate all the nodes in the given DOMTree.
            for node in dom_tree.iter():
                # The value can only be matched in the text of the node or the tail.
                node_text_dict = {"node_text": node.text, "node_tail_text": node.tail}
                
                for text_part_flag, node_text in node_text_dict.items():
                    if node_text:
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

                        len_brs = len(node_text.split("--BRRB--"))  # The number of the <br>s.
                        for index, etext in enumerate(node_text.split("--BRRB--")):
                            if text_part_flag == "node_text":
                                xpath = dom_tree.getpath(node)
                            elif text_part_flag == "node_tail_text":
                                # TODO (aimore): I am not sure why they make the distinction of the
                                #  xpath being the text or tail. That increases complexity of xpath and
                                #  might not help during generalization.
                                xpath = dom_tree.getpath(node) + "/tail"
                            if len_brs >= 2:
                                xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]
                            clean_etext = clean_spaces(etext)

                            # Update the dictionary.
                            current_xpath_data[xpath] = clean_etext
                            overall_xpath_dict[xpath].add(clean_etext)
                            current_page_nodes_in_order.append((clean_etext, xpath))

                            # Clean the groundtruth and the node text. Check if the groundtruth is in the node text.                        
                            clean_etext = clean_format_str(clean_etext)

                            for gt_value in clean_gt_values:
                                if gt_value.strip() in clean_etext.strip():
                                    matched_xpaths.append(xpath)
                                    is_truth_value_list.append(len(current_page_nodes_in_order) - 1)
                                    break

            # Update the page-level xpath information.
            all_data_dict[page_id][field]["groundtruth_xpaths"].update(matched_xpaths)
            all_data_dict[page_id][field]["is_truth_value_list"].update(is_truth_value_list)

            # now for each all_data_dict[page_id]
            # an example
            # all_data_dict[page_id]["field-PAST_CLIENT"] =
            # {
            #   'values': ['Dave Kemper', 'Patrick Sebranek', 'Verne Meyer'],
            #   'groundtruth_xpaths':
            #       {'/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[3]',
            #        '/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[2]',
            #        '/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[1]',
            #        '/html/body/div[2]/div[2]/div[3]/div[3]/p/a'}
            # }

        all_data_dict[page_id]["xpath_data"] = current_xpath_data  # {xpath1: text1, xpath2: text2}
        all_data_dict[page_id]["doc_strings"] = current_page_nodes_in_order  # [(text, xpath)*N]
        # all_data_dict[page_id]["reversed_doc_strings_ids"] = {v[0]: i for i, v in enumerate(current_page_nodes_in_order)}

    # all_data_dict[page_id]["doc_strings"] is the basis of our transformers-based method!!!

    # Define the fixed-text nodes and variable nodes.
    fixed_nodes = set()
    variable_nodes = set(overall_xpath_dict.keys())

    assert len(fixed_nodes) == 0 
    assert len(variable_nodes) > 0 
    print(f"Website: {website_to_process} | Across all pages:\n \tfixed_nodes: {len(fixed_nodes)} | \tvariable_nodes: {len(variable_nodes)}")

    assure_value_variable(all_data_dict, variable_nodes, fixed_nodes)
    all_data_dict["fixed_nodes"] = list(fixed_nodes)
    all_data_dict["variable_nodes"] = list(variable_nodes)

    # 总之到这为止 (Anyway so far)
    # fixed_nodes包含的就是固定的node (fixed_nodes contains fixed nodes)
    # variable_nodes包含的就是值会变化的node (variable_nodes contains nodes whose values will change)
    # 并且我们保证truth_value必定在variable nodes中
    # (and we guarantee that truth_value must be in variable nodes)

    # "fixed_nodes" are the xpaths for nodes that cannot have truth-value
    # "variable_nodes" are the xpaths for nodes that might have truth-value
    return all_data_dict 


def assure_value_variable(all_data_dict, variable_nodes, fixed_nodes):
    """Makes sure all values are in the variable nodes by updating sets.
    
    That means that if the xpath that is in groundtruth_xpaths was not yet in the variable_nodes, then the variable_nodes gets updated with it.
    And it gets removed from fixed-xpath.

    Args:
      all_data_dict: the dictionary saving all data with groundtruth.
      variable_nodes: the current set of variable nodes.
      fixed_nodes: the current set of fixed nodes.
      n_pages: to assume we only process first n_pages pages from each website.
    """
    for index in all_data_dict:
        if not index.isdigit():
            # the key should be an integer, to exclude "fixed/variable nodes" entries.
            # n_pages to stop for only process part of the website.
            continue
        for field in all_data_dict[index]:
            if not field.startswith("field-"):
                continue
            xpaths = all_data_dict[index][field]["groundtruth_xpaths"]
            if not xpaths:  # There are zero value for this field in this page.
                continue
            flag = False
            for xpath in xpaths:
                if flag:  # The value's xpath is in the variable_nodes.
                    break
                flag = xpath in variable_nodes
            variable_nodes.update(xpaths)  # Add new xpaths if they are not in.
            fixed_nodes.difference_update(xpaths)


def generate_nodes_seq_and_write_to_file(website):
    """Extracts all the xpaths and labels the nodes for all the pages."""

    all_data_dict = load_html_and_groundtruth(website)
    """
    all_data_dict = {'0000': {
    field-PAST_CLIENT: {'values': ['we work', 'sse', 'Oman Investment Corporation']}, 
    'path':'addleshawgoddard.com(105)/0000.htm', 
    'html_str':'html' }, 
    '0001': {...} }}
    """

    all_data_dict = get_field_xpaths(
        all_data_dict=all_data_dict,
        website_to_process=website,
    )

    """
    all_data_dict = {'0000': {
    field-PAST_CLIENT: 
        {'values': ['SA', 'Luye Pharma Group Ltd.', 'Vipshop (US) Inc', 'Delta Air Lines', 'New York University', 'Harmon Store', 'Rutgers University', 'Amneal Pharmaceuticals, LLC', 'Kashiv Pharma, LLC', ...], 
         'groundtruth_xpaths': {'/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[11]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[12]/h2', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[21]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[19]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[3]/h2', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[20]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[10]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[3]/p[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[18]/p[1]', ...}, 
         'is_truth_value_list': {384, 256, 128, 649, 533, 410, 412, 669, 555, ...}
         }, 
    'path': 'lernerdavid.com(41)/0000.htm', 
    'html_str': 'html' }, 
    'dom_tree': dom_tree
    'xpath_data': {'/html/head': '', '/html/head/script[1]/tail': '', '/html/head/script[2]': 'window.jQuery || document.write(\'<script src="/Darwin/script/jquery/jquery-1.11.2.min.js"><\\/script>\')', '/html/head/script[2]/tail': '', '/html/head/script[3]/tail': '', '/html/head/script[4]': 'window.jQuery.ui || document.write(\'<script src="/Darwin/script/jquery/jquery-ui-1.12.1.min.js"><\\/script>\')', '/html/head/script[4]/tail': '', '/html/head/title': 'Successes', '/html/body': '', '/html/body/noscript[1]/tail': '', '/html/body/noscript[1]/div': 'Javascript must be enabled for the correct page display', '/html/body/script[1]': "//<![CDATA[var theForm = document.forms['aspnetForm'];if (!theForm) { theForm = document.aspnetForm;}function __doPostBack(eventTarget, eventArgument) { if (!theForm.onsubmit || (theForm.onsubmit() != false)) { theForm.__EVENTTARGET.value = eventTarget; theForm.__EVENTARGUMENT.value = eventArgument; theForm.submit(); }}//]]>", '/html/body/script[1]/tail': '', '/html/body/div[2]': '', ...}
    'doc_strings': [('', '/html/head'), ('', '/html/head/script[1]/tail'), ('window.jQuery || document.write(\'<script src="/Darwin/script/jquery/jquery-1.11.2.min.js"><\\/script>\')', '/html/head/script[2]'), ('', '/html/head/script[2]/tail'), ('', '/html/head/script[3]/tail'), ('window.jQuery.ui || document.write(\'<script src="/Darwin/script/jquery/jquery-ui-1.12.1.min.js"><\\/script>\')', '/html/head/script[4]'), ('', '/html/head/script[4]/tail'), ('Successes', '/html/head/title'), ('', '/html/body'), ('', '/html/body/noscript[1]/tail'), ('Javascript must be enabled for the correct page display', '/html/body/noscript[1]/div'), ("//<![CDATA[var theForm = document.forms['aspnetForm'];if (!theForm) { theForm = document.aspnetForm;}function __doPostBack(eventTarget, eventArgument) { if (!theForm.onsubmit || (theForm.onsubmit() != false)) { theForm.__EVENTTARGET.value = eventTarget; theForm.__EVENTARGUMENT.value = eventArgument; theForm.submit(); }}//]]>", '/html/body/script[1]'), ('', '/html/body/script[1]/tail'), ('', '/html/body/div[2]'), ...]
    '0001': {...} }},
    'fixed_nodes': [],
    'variable_nodes': ['/html/body/div[2]/div/div/header/div/div/nav/div/ul/li[3]/ul/li[1]/a', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[24]/dl[1]/tail', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[24]/p[1]/tail', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[12]/h2', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[17]/dl[2]/dt/tail', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[11]/dl[1]/dt', '/html/body/div[2]/div/div/div[4]/div/aside/div/ul/li/tail', '/html/body/div[2]/div/div/div[4]/div/section/tail', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[18]/dl[2]/tail', '/html/head/title', '/html/body/div[2]/div/div/div[4]/div/aside/ul[23]/tail', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[19]/dl[1]/dd[2]/a', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[9]/dl[1]', '/html/body/div[2]/div/div/div[4]/div/section/div[1]/article[1]/dl[2]', ...]
    """

    variable_nodes = all_data_dict["variable_nodes"]

    cleaned_features_for_this_website = {}

    page_ids = [keys for keys in all_data_dict if 'nodes' not in keys] # The keys of all_data_dict include also fixed_nodes and variable_nodes and here we only care about the page_ids
    for page_id in page_ids:
        page_data = all_data_dict[page_id]

        assert "xpath_data" in page_data

        doc_strings = page_data["doc_strings"]
        # E.g. doc_strings = [('1730 Pennsylvania Avenue NW | HITT', '/html/head/title'), ...]
        new_doc_strings = []
        # The difference between doc_strings and new_doc_strings is that new_doc_strings contains the label
        field_info = {}
        # TODO(Aimore): Probably change the name of this variable (field_info) to something more meaningful, such as: gt_node_ids
        for field in [x for x in list(page_data.keys()) if 'field' in x]:
            for doc_string_id in page_data[field]["is_truth_value_list"]:
                field_info[doc_string_id] = field.split("-")[1]

        for id, doc_string in enumerate(doc_strings):
            text, xpath = doc_string
            is_variable = xpath in variable_nodes
            # Define Fixed-nodes
            if not is_variable:
                new_doc_strings.append((text, xpath, "fixed-node"))
            # Define Variable-nodes
            else:
                gt_field = field_info.get(id, "none")  # Choose between none or gt label (PAST_CLIENT)
                new_doc_strings.append((text, xpath, gt_field))

        cleaned_features_for_this_website[page_id] = new_doc_strings

    output_file_path = os.path.join(FLAGS.output_data_path, f"{website}.pickle")
    print(f"Writing the processed and {len(cleaned_features_for_this_website)} pages of {website} into {output_file_path}")
    nodes_per_page = {x[0]:len(x[1]) for x in cleaned_features_for_this_website.items()}
    print(f"# Nodes per page: {nodes_per_page}\n")
    with open(output_file_path, "wb") as f:
        pickle.dump(cleaned_features_for_this_website, f)


def main(_):
    if not os.path.exists(FLAGS.output_data_path):
        os.makedirs(FLAGS.output_data_path)

    swde_path = FLAGS.input_groundtruth_path.split("groundtruth")[0]
    p = Path(swde_path) / "WAE"
    websites = sorted([x.parts[-1].split("-")[-1].split("(")[0] for x in list(p.iterdir())])

    # websites = [x for x in websites if "ciphr.com" not in x] # TODO: Remove this website for now just because it is taking too long (+20min.) 

    print(f"Prepare Data: Websites {len(websites)} -\n {websites}")

    # for website in websites[:]:
    #     # if website in ['lernerdavid.com', 'awarehq.com', 'addleshagoddard.com']:
    #     # if website in ['addleshawgoddard.com']:
    #     if 'direct.com' in website: # TODO (AIMORE): Debug this page and understand why it doesn't have variable nodes
    #         generate_nodes_seq_and_write_to_file(website)

    # from p_tqdm import p_uimap
    # iterator = p_uimap(generate_nodes_seq_and_write_to_file, websites)
    # for e, result in enumerate(iterator):
    #     print(e)

    num_cores = mp.cpu_count()
    with mp.Pool(num_cores) as pool, tqdm(total=len(websites), desc="Processing swde-data") as t:
        # for res in pool.imap_unordered(generate_nodes_seq_and_write_to_file, websites):
        for res in pool.imap(generate_nodes_seq_and_write_to_file, websites):
            t.update()


if __name__ == "__main__":
    app.run(main)
