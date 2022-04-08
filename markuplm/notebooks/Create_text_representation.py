# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.8.12 ('wae_test')
#     language: python
#     name: python3
# ---

# +
from pathlib import Path
import pandas as pd

pd.set_option("max_colwidth", 200, "max_rows", 10, "min_rows", 10)
# -

# # Load Data

results_df = pd.read_pickle("results_classified_5_epoch.pkl")
initial_node_count = len(results_df)
print(f"Df result size: {initial_node_count}")
results_df.head(4)

results_df_pos_more_than_1000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] > 1000)])
results_df_pos_less_than_1000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] < 1000)])
print(f"results_df_pos_more_than_1000: {results_df_pos_more_than_1000}\nresults_df_pos_less_than_1000:{results_df_pos_less_than_1000}")

# +

# #? Load and apply pageid to url mapping
pageid_url_mapping = pd.read_pickle("/data/GIT/swde/my_data/develop/my_CF_sourceCode/pageid_url_mapping.pkl")
results_df.reset_index(inplace=True)
results_df = results_df.drop("index", axis=1)
results_df['url'] = results_df['html_path'].apply(lambda x: pageid_url_mapping.get(x)[0])

# #? Clean domain name
results_df['domain'] = results_df['domain'].apply(lambda x: x.split(".pickle")[0])

# +
from lxml.html.clean import Cleaner

cleaner = Cleaner()
cleaner.forms = True
cleaner.annoying_tags = True
cleaner.page_structure = True
cleaner.inline_style = True
cleaner.scripts = True
cleaner.javascript = True # This is True because we want to activate the javascript filter

def clean_node(text):
    try:
        return cleaner.clean_html(text)[3:-4]
    except:
        pass

min_char = 1
max_char = 1000

# #? Clean node text 
results_df['text'] = results_df['text'].apply(clean_node) 
results_df['node_text_len'] = results_df["text"].dropna().apply(len)

print(f"Df result size: {len(initial_node_count)}")

results_df_pos_more_than_1000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] > 1000)])
results_df_pos_less_than_1000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] < 1000)])
print(f"results_df_pos_more_than_1000: {results_df_pos_more_than_1000}\nresults_df_pos_less_than_1000:{results_df_pos_less_than_1000}")

results_df = results_df[results_df['node_text_len'] >= min_char] 
print(f"Df result size: {len(results_df)} - Filter out nodes with text smaller or equal than {min_char} characters")
results_df = results_df[results_df["node_text_len"] < max_char] 
print(f"Df result size: {len(results_df)} - Filter out nodes with text longer than {max_char} characters")
# -

# # Generate text for html

# +
from typing import List

node_show = False
node_text_link = False

def make_bold_gt_texts(text:str, gt_texts:List[str]) -> str:
    for gt_text in gt_texts:
        text = text.replace(gt_text, f"<b>{gt_text}</b>")
    return text

# #? Create folder
def create_folder(folder_path="text_representation/"):
    text_representation_folder = Path(folder_path)
    if text_representation_folder.exists():
        text_representation_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {text_representation_folder} folder.")
create_folder()

def define_page(url, url_df, url_id):
    node_text_list = []
    for index_node, (_, node) in enumerate(url_df.iterrows()):
        node_defined = define_node(index_node, node, url)
        node_text_list.extend(node_defined)

    nodes_text = "".join(node_text_list)

    collapse = "collapse"
    if "<p class='TP'>" in nodes_text or "<p class='FP'>" in nodes_text or "<p class='FN'>" in nodes_text:
        collapse = "show"

    page_title_link = f"<a href={url}>{url}</a>"
    page_template = f"""
        <div class="panel panel-default">
            <a data-toggle="collapse" href="#collapse{url_id}">>>{url_id}<<</a> {page_title_link}
            <div id="collapse{url_id}" class="panel-collapse {collapse}">
                <div class="panel-body">{nodes_text}</div>
            </div>
        </div>
    """
    return page_template
    
# #? Define what nodes are TP/FP/FN
def define_node(index_node, df_node, url):
    xpath = df_node['xpath']
    text = df_node['text']
    gt_texts = df_node['gt_text']

    node_index = str(index_node)
    if len(gt_texts) > 0:
        text = make_bold_gt_texts(text, gt_texts)

    if node_show:
        text = f"<p class='xpath'> {node_index}: {xpath}</p>" + text

    if node_text_link:
        text_ref = f"{url}#:~:text={text.strip().replace(' ', '%20')}"
        text_ref_link = f"<a href={text_ref}>{text}</a>"
        text = text_ref_link

    if df_node['truth'] == 'PAST_CLIENT' and df_node['pred_type'] == 'PAST_CLIENT':
        text_to_return = f"<p class='TP'> {text} </p>"

    elif df_node['truth'] == 'PAST_CLIENT' and df_node['pred_type'] == 'none':
        text_to_return = f"<p class='FN'> {text} </p>"

    elif df_node['truth'] == 'none' and df_node['pred_type'] == 'PAST_CLIENT':
        text_to_return = f"<p class='FP'> {text} </p>"
    else:
        text_to_return = f"<p> {text} </p>"

    # return f"<div>{text_to_return}</div>\n"
    return f"{text_to_return}\n"

# #? Create the text representation of the html
def create_text_representation_for_website(website, website_df, folder_path="text_representation"):
    pages_list = []

    for url_id, (url, url_df) in enumerate(website_df.groupby('url')):
        print(f"\t{url_id}: {url}")
        page_defined = define_page(url, url_df, url_id)
        pages_list.extend(page_defined)

    full_text = ''.join(pages_list)

    html_text = f"""
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
    </head>

    <link rel='stylesheet' href='styles.css'>
    <!DOCTYPE html>
    <html> {full_text} </html>
    """
    
    # html_text = re.sub("&(?!amp;)", "&amp;", html_text)
    # TODO (AIMORE): Apply these preprocessing before but also when predicting!
    html_text = html_text.replace("&amp;", "&")
    html_text = html_text.replace("&AMP;", "&")
    with open(f'{folder_path}/{website}.html', 'w') as f:
        f.write(html_text)

# #? Run through the websites
for website_id, (website, df_website) in enumerate(results_df.groupby('domain')):
    if website == "piwik.pro":
        print(f"{website_id}: {website}")
        create_text_representation_for_website(website, df_website)
        # break
    # if website_id > 10:
    #     break

# TODO: Add colourful name of the Past Clients 
# TODO: Add link to the found sentence {url}#:~:text=%20the%20
# -

# # Error analysis

# +
import re2 as re

re.sub("&(?!amp;)", "&amp;", html_text)

# +
# develop_df = pd.read_pickle(f"/data/GIT/web-annotation-extractor/data/processed/develop/dataset_pos(1735)_neg(4035)_intermediate.pkl")
# develop_df
# -




