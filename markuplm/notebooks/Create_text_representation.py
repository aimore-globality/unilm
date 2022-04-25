# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.8.12 ('wae_test')
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import pandas as pd

# pd.set_option("max_colwidth", 200, "max_rows", 10, "min_rows", 10)

# %% [markdown]
# # Load Data

# %%
# results_df = pd.read_pickle("results_classified/results_classified_5_epoch.pkl")
# results_df = pd.read_pickle("results_classified/develop_set_nodes_classified_epoch_3.pkl")
results_df = pd.read_pickle("results_classified/develop_set_nodes_classified_epoch_1.pkl")


initial_node_count = len(results_df)
print(f"Df result size: {initial_node_count}")
results_df.head(4)

# %%
# results_df["node_text_len"] = results_df["text"].apply(len)
# results_df_pos_more_than_1000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] > 1000)])
# results_df_pos_less_than_1000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] < 1000)])
# print(f"results_df_pos_more_than_1000: {results_df_pos_more_than_1000}\nresults_df_pos_less_than_1000:{results_df_pos_less_than_1000}")

# %%

# #? Load and apply pageid to url mapping
pageid_url_mapping = pd.read_pickle("/data/GIT/swde/my_data/develop/my_CF_sourceCode/pageid_url_mapping.pkl")
results_df.reset_index(inplace=True)
results_df = results_df.drop("index", axis=1)
results_df['url'] = results_df['html_path'].apply(lambda x: pageid_url_mapping.get(x)[0])

# #? Get domain name from html_path
results_df['domain'] = results_df['html_path'].apply(lambda x: x.split(".pickle")[0])

# %%
# [x for x in pd.DataFrame(results_df.groupby('domain'))[0].values]

# %%
seed = 66 
websites_for_error_analysis = sorted(pd.DataFrame(results_df.groupby('domain'))[0].sample(frac=0.1, random_state=seed).values)
print(websites_for_error_analysis)

# %%

# #? Interesting analysis if we remove the nodes with duplicated data, we can massively reduce their size.
duplicated_nodes = results_df
domain_non_duplicated_nodes = results_df.drop_duplicates(subset=["text", "domain"])
print(f"{'All nodes:':>50} {len(duplicated_nodes):>7}")
print(f"{'Domain non-duplicated nodes:':>50} {len(domain_non_duplicated_nodes):>7} ({100*len(domain_non_duplicated_nodes)/len(duplicated_nodes):.2f} %)")

# #? Also, not so many nodes with positive data are removed compared to the other data.
duplicated_gt = len(duplicated_nodes[duplicated_nodes["truth"] != 'none'])
domain_non_duplicated_gt = len(domain_non_duplicated_nodes[domain_non_duplicated_nodes["truth"] != 'none'])
print(f"{'All number of ground truth nodes:':>50} {duplicated_gt:>7}")
print(f"{'Domain non duplicated ground truth nodes:':>50} {domain_non_duplicated_gt:>7} ({100*(domain_non_duplicated_gt) / duplicated_gt:.2f} %)")

# %%
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
max_char = 10_000

# #? Clean node text 
results_df['text'] = results_df['text'].apply(clean_node) 
results_df['node_text_len'] = results_df["text"].dropna().apply(len)

print(f"Df result size: {initial_node_count}")

results_df_pos_more_than_10000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] > 10000)])
results_df_pos_less_than_10000 = len(results_df[(results_df["truth"] != "none") & (results_df["node_text_len"] < 10000)])
print(f"results_df_pos_more_than_10000: {results_df_pos_more_than_10000}\nresults_df_pos_less_than_10000:{results_df_pos_less_than_10000}")

# %% [markdown]
# # Generate text for html

# %%
from typing import List

show_node = False
node_text_link = False
show_probability = True
show_node_tag = True

def make_bold_gt_texts(text:str, gt_texts:List[str]) -> str:
    for gt_text in gt_texts:
        text = text.replace(gt_text, f"<b>{gt_text}</b>")
    return text

# #? Create folder
def create_folder(folder_path="text_representation/"):
    text_representation_folder = Path(folder_path)
    if text_representation_folder.exists():
        print(f"{text_representation_folder} folder already exists.")
    else:
        text_representation_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {text_representation_folder} folder.")
create_folder()

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
    text = df_node['text']
    node_tag = df_node['node_tag']
    gt_texts = df_node['gt_text']
    node_prob = df_node['final_probs'][0]
    node_index = str(index_node)            


    if len(gt_texts) > 0:
        text = make_bold_gt_texts(text, gt_texts)

    if show_node:
        text = f"<p class='xpath'> {node_index}: {xpath}</p>" + text
        text = f"<div class='column'> {text} </div>\n"

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

    text_to_return =f"<div class='column first'> {text_to_return} </div>"

    if show_probability:
        text_to_return = f"<div class='column second'> <p>{node_prob:.2f}</p> </div> {text_to_return}\n"

    if show_node_tag:
        text_to_return = f"<div class='column third'> <p>{node_tag}</p> </div> {text_to_return}\n"

    text_to_return = f"<div class='row'> {text_to_return}</div>\n"

    return text_to_return

# #? Create the text representation of the html
def create_text_representation_for_website(website, website_df, folder_path="text_representation"):
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
    width: 95%;
    }

    .second {
    width: 2%;
    }
    
    .third {
    width: 3%;
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
    with open(f'{folder_path}/{website}.html', 'w') as f:
        f.write(html_text)

# #? Run through the websites
for website_id, (website, df_website) in enumerate(results_df.groupby('domain')):
    # if website == "lssmedia.com":
    #     print(f"{website_id}: {website}")
    create_text_representation_for_website(website, df_website)
    # break
    # if website_id > 1:
    #     break

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
