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
from bs4 import BeautifulSoup

pd.set_option("max_colwidth", 200, "max_rows", 10, "min_rows", 10)

# %% [markdown]
# # Load Data

# %%
results_df = pd.read_pickle("results_classified_5_epoch.pkl")
print(f"Df result size: {len(results_df)}")

# %%

# #? Load and apply pageid to url mapping
pageid_url_mapping = pd.read_pickle("/data/GIT/swde/my_data/develop/my_CF_sourceCode/pageid_url_mapping.pkl")
results_df.reset_index(inplace=True)
results_df = results_df.drop("index", axis=1)
results_df['url'] = results_df['html_path'].apply(lambda x: pageid_url_mapping.get(x)[0])

# #? Clean domain name
results_df['domain'] = results_df['domain'].apply(lambda x: x.split(".pickle")[0])

# %%
from lxml.html.clean import Cleaner

cleaner = Cleaner()
cleaner.forms = True
cleaner.annoying_tags = True
cleaner.page_structure = True
cleaner.inline_style = True
cleaner.scripts = True
cleaner.javascript = True # This is True because we want to activate the javascript filter

# def clean_node(text):
#     try:
#         return BeautifulSoup(cleaner.clean_html(text), "lxml").text
#     except:
        # pass

def clean_node(text):
    try:
        return cleaner.clean_html(text)
    except:
        pass

# TODO(Aimore): It seems that there are 
min_char = 1
max_char = 1000

print(f"Df result size: {len(results_df)}")
results_df = results_df[results_df['node_text_len'] >= min_char] 
print(f"Df result size: {len(results_df)} - Filter out nodes with text smaller or equal than {min_char} characters")
results_df = results_df[results_df["node_text_len"] < max_char]
print(f"Df result size: {len(results_df)} - Filter out nodes with text longer than {max_char} characters")

# #? Clean node text 
results_df['text'] = results_df['text'].apply(clean_node) 
results_df['node_text_len'] = results_df["text"].dropna().apply(len)

results_df = results_df[results_df['node_text_len'] >= min_char] 
print(f"Df result size: {len(results_df)} - Filter out nodes with text smaller or equal than {min_char} characters")
results_df = results_df[results_df["node_text_len"] < max_char] 
print(f"Df result size: {len(results_df)} - Filter out nodes with text longer than {max_char} characters")


# %% [markdown]
# # Genereate text for html

# %%

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
        node_defined = define_node(index_node, node)
        node_text_list.extend(node_defined)

    nodes_text = "".join(node_text_list)

    page_title = f"<a href={url}>{url}</a>"
    page_template = f"""
        <div class="panel panel-default">
            <a data-toggle="collapse" href="#collapse{url_id}">>>{url_id}<<</a> {page_title}
            <div id="collapse{url_id}" class="panel-collapse">
                <div class="panel-body">{nodes_text}</div>
            </div>
        </div>
    """
    return page_template
    

# #? Define what nodes are TP/FP/FN
def define_node(index_node, df_node):
    xpath = df_node['xpath']
    text = df_node['text']
    text = clean_html(text)
    node_index = str(index_node)

    # text = .replace('script', 'AIMORESCRIPT') #! Check if there is a better way to remove scripts
    # text = f"<p class='xpath'> {node_index}: {xpath}</p>" + text

    if df_node['truth'] == 'PAST_CLIENT' and df_node['pred_type'] == 'PAST_CLIENT':
        text_to_return = f"<e class='TP'> {text} </e>"

    elif df_node['truth'] == 'PAST_CLIENT' and df_node['pred_type'] == 'none':
        text_to_return = f"<e class='FN'> {text} </e>"

    elif df_node['truth'] == 'none' and df_node['pred_type'] == 'PAST_CLIENT':
        text_to_return = f"<e class='FP'> {text} </e>"
    else:
        text_to_return = f"<e> {text} </e>"

    # return f"<div>{text_to_return}</div>\n"
    return f"{text_to_return}\n"

# #? Create the text representation of the html
def create_text_representation_for_website(website, website_df, folder_path="text_representation"):
    pages_list = []

    for url_id, (url, url_df) in enumerate(website_df.groupby('url')):
        print(f"\t{url_id}: {url}")
        page_defined = define_page(url, url_df, url_id)
        pages_list.extend(page_defined)

    # TODO: 1. Add division for pages - collapsable
    # 2. Add name of node 
    # 3. Add name (link) of the url 
    # 4. Add colourful name of the Past Clients 

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

    with open(f'{folder_path}/{website}.html', 'w') as f:
        f.write(html_text)

# #? Run through the websites
for website_id, (website, df_website) in enumerate(results_df.groupby('domain')):
    print(f"{website_id}: {website}")
    create_text_representation_for_website(website, df_website)
    
    if website_id == 0:
        break

# %%
