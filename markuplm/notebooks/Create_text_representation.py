# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.7.11 ('markuplmft')
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import pandas as pd
import glob 

pd.set_option("max_colwidth", 50, "max_rows", 4, "min_rows", 4)

# %% [markdown]
# # Generate text for html

# %%
# df_website.head(1)

# %%
# df_website["gt_tag"].value_counts()

# %%
from typing import List

show_node = True
node_text_link = False
show_probability = True
show_node_gt_tag = True

def make_bold_gt_texts(text:str, node_gt_text:List[str]) -> str:
    for gt_text in node_gt_text:
        text = f" {text} ".replace(f" {gt_text} ", f" <b>{gt_text}</b> ")
    return text

def create_folder(folder_path):
    text_representation_folder = Path(folder_path)
    if text_representation_folder.exists():
        print(f"{text_representation_folder} folder already exists.")
    else:
        text_representation_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {text_representation_folder} folder.")

def define_page(url, url_df, url_id):
    node_text_list = []
    for index_node, (_, node) in enumerate(url_df.iterrows()):
        # if index_node == 10: # ? For debugging specific Node
        #     print(f"Reached node: {index_node}")
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
    text = df_node['node_text']
    node_gt_tag = df_node['node_gt_tag']
    node_gt_text = df_node['node_gt_text']
    node_pred_tag = df_node['node_pred_tag']

    if 'node_prob' in list(df_node.index):
        node_prob = df_node['node_prob'][0]
    else:
        node_prob = 0

    if 'node_pred_tag' not in list(df_node.index):
        node_pred_tag = 'none'
            
    node_index = str(index_node)

    if len(node_gt_text) > 0:
        text = make_bold_gt_texts(text, node_gt_text)

    if node_text_link:
        text_ref = f"{url}#:~:text={text.strip().replace(' ', '%20')}"
        text_ref_link = f"<a href={text_ref}>{text}</a>"
        text = text_ref_link

    if node_gt_tag == 'PAST_CLIENT' and node_pred_tag == 'PAST_CLIENT':
        text_to_return = f"<p class='TP'> {text} </p>"

    elif node_gt_tag == 'PAST_CLIENT' and node_pred_tag == 'none':
        text_to_return = f"<p class='FN'> {text} </p>"

    elif node_gt_tag == 'none' and node_pred_tag == 'PAST_CLIENT':
        text_to_return = f"<p class='FP'> {text} </p>"
    else:
        text_to_return = f"<p> {text} </p>"

    text_to_return =f"<div class='column first'> {text_to_return} </div>"

    if show_probability:
        text_to_return = f"<div class='column second'> <p>{node_prob:.2f}</p> </div> {text_to_return}\n"

    if show_node_gt_tag:
        text_to_return = f"<div class='column third'> <p>{node_gt_tag}</p> </div> {text_to_return}\n"

    text_to_return = f"<div class='row two'> {text_to_return}</div>\n"

    if show_node:
        text_to_return = f"<div class='row one'> <p2>{node_index}: {xpath}</p2> </div> {text_to_return}\n"
    
    return text_to_return

# #? Create the text representation of the html
def create_text_representation_for_website(website, website_df, folder_path):
    pages_list = []

    for url_id, (url, url_df) in enumerate(website_df.groupby('url')):
        print(f"\t{url_id}: {url}")
        page_defined = define_page(url, url_df, url_id)
        pages_list.extend(page_defined)

    full_text = ''.join(pages_list)


    head_style = """
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    * {
        box-sizing: border-box;
    }

    .TP {
    background-color: rgba(9, 255, 0, 0.5);
    }
    .FN {
        background-color: rgba(255, 187, 0, 0.7);
    }
    .FP {
        background-color: rgba(255, 53, 10, 0.5);
    }
    p {
    font-size:10px;
    margin-bottom: -1px;
    /* background-color: rgba(226, 53, 10, 0.877); */
    /* padding: 0; */
    /* white-space: 0; */
    /* line-height: 0; */
    }

    p2 {
    font-size:9px;
    line-height: 25px;
    margin-bottom: -10px;
    /* background-color: rgba(226, 53, 10, 0.877); */
    /* padding: 0; */
    /* white-space: 0; */
    /* line-height: 0; */
    }
    
    .row {
        display: flex;
        flex: 50%;
        padding: 1px;
    }

    /* Create two equal columns that sits next to each other */
    .column {
        padding: 1px;
    }
    .first {
    width: 90%;
    }
    .second {
    width: 4%;
    }
    .third {
    width: 6%;
    }
    

    </style>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
    </head>

    <link rel='stylesheet' href='styles.css'>
    <!DOCTYPE html>
    """

    html_text = f"{head_style}\n<html> {full_text} </html>"
    
    # html_text = re.sub("&(?!amp;)", "&amp;", html_text)
    # TODO (AIMORE): Apply these preprocessing before but also when predicting!
    html_text = html_text.replace("&amp;", "&")
    html_text = html_text.replace("&AMP;", "&")
    save_path = folder_path / f"{website}.html"
    print(f"Saved at: {save_path}")
    with open(save_path, 'w') as f:
        f.write(html_text)

dataset = "develop"
load_folder_path = Path(f"/data/GIT/delete/{dataset}/processed")
save_folder_path = Path(f"/data/GIT/delete/{dataset}/text_representation")
create_folder(save_folder_path)
domains_path = glob.glob(str(load_folder_path / "*.pkl"))
# domains_path = glob.glob(f"/data/GIT/delete/{dataset}/processed_dedup/*.pkl")


for website_id, website_path in enumerate(domains_path):
    website = website_path.split('/')[-1]
    print(f"{website_id}: {website}")
    df_website = pd.read_pickle(website_path)
    df_website = df_website.explode("nodes", ignore_index=True).reset_index()
    df_website = df_website.join(pd.DataFrame(df_website.pop('nodes').tolist(), columns=["xpath","node_text","node_gt_tag","node_gt_text"]))
    
    create_text_representation_for_website(website, df_website, save_folder_path)
    break

# TODO: Add colourful name of the Past Clients 
# TODO: Add link to the found sentence {url}#:~:text=%20the%20

# %% [markdown]
# # Error analysis

# %% [markdown]
# ## Copy domain text_representation of a sample to folder to be analysed

# %%
import shutil

src = Path("text_representation")
dst = Path("erro_analysis_domains")
if not dst.exists():
    dst.mkdir()

for website, data in results_df[results_df["domain"].isin(websites_for_error_analysis)].groupby('domain'):
    print(f"{website}: {len(data[data['truth']!='none'])}")
    website = website + '.html'

    shutil.copyfile(src / website, dst / website)
