# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.9.12 ('wae39_wiki')
#     language: python
#     name: python3
# ---

# %%
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.common.utils.parallel_tools import OptimalParallel
from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
from web_annotation_extractor.common.utils.general_utils import deserialize_annotations
import re2 as re
import pandavro as pdx
from pathlib import Path
from ast import literal_eval
import numpy as np
import pandas as pd
import unidecode
import wikipedia

import string
from marquez.enums.annotation import EntityTag
graph = create_object_graph('test')

# %%
graph = create_object_graph("gazetteers")

known_company_taxonomy = []
for company in graph.known_company_taxonomy:
    if company.is_demo_company is False and company.deprecated is False:
        known_company_taxonomy.append(company)

company_name_to_uri_map = dict({
    (company.name, company.uri)
    for company in known_company_taxonomy
})
uri_to_company_name_map = dict({
    (company.uri, company.name)
    for company in known_company_taxonomy
})

known_company_names = pd.DataFrame([company.name for company in known_company_taxonomy])
known_company_names_taxonomy = pd.DataFrame([(company, company.name) for company in known_company_taxonomy], columns=["taxonomy_id", "company_name"])

# %%
from collections import OrderedDict

def get_information(company_name, verbose=False):
    info = OrderedDict()

    print(f"Company Name: {company_name}")
    print(f"Searched: {search_expression }")
    search_expression = f"{company_name} company"    

    wiki_page = wikipedia.page(search_expression)
    info["url"] = wiki_page.url
    info["title"] = wiki_page.title
    info["categories"] = wiki_page.categories
    info["summary"] = wiki_page.summary
    info["content"] = wiki_page.content
    if verbose:
        [print(f"{key}: {values}") for (key,values) in info.items()]
    return info


# %%
all_wiki_page_info = []
for company_name in known_company_names_taxonomy['company_name'][:10]:
    wiki_page_info = get_information(company_name, verbose=False)
    all_wiki_page_info.append(wiki_page_info)
    print("-"*100)

# %%
known_company_names_taxonomy_extended = known_company_names_taxonomy.join(pd.DataFrame(all_wiki_page_info))

# %%
known_company_names_taxonomy_extended[["taxonomy_id", "company_name", "title", "url", "summary"]].dropna()

# %%
print()
# Wikipedia (/ˌwɪkɨˈpiːdiə/ or /ˌwɪkiˈpiːdiə/ WIK-i-PEE-dee-ə) is a collaboratively edited, multilingual, free Internet encyclopedia supported by the non-profit Wikimedia Foundation...

wikipedia.search("Barack")
# [u'Barak (given name)', u'Barack Obama', u'Barack (brandy)', u'Presidency of Barack Obama', u'Family of Barack Obama', u'First inauguration of Barack Obama', u'Barack Obama presidential campaign, 2008', u'Barack Obama, Sr.', u'Barack Obama citizenship conspiracy theories', u'Presidential transition of Barack Obama']

ny = wikipedia.page("New York")
ny.title
# u'New York'
ny.url
# u'http://en.wikipedia.org/wiki/New_York'
ny.content
# u'New York is a state in the Northeastern region of the United States. New York is the 27th-most exten'...
ny.links[0]
# u'1790 United States Census'

wikipedia.set_lang("fr")
wikipedia.summary("Facebook", sentences=1)
# Facebook est un service de réseautage social en ligne sur Internet permettant d'y publier des informations (photographies, liens, textes, etc.) en contrôlant leur visibilité par différentes catégories de personnes.
