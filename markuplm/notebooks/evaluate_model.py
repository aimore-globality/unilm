# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %%
from markuplmft.fine_tuning.run_swde.lm_model import LModel
import pandas as pd

lm = LModel()
lm.load_data('develop')
results = lm.predict_on_develop()

# %%
lm.evaluate(results)

# %%
results = results[results['node_text_len'] > 1]
lm.evaluate(results)

# %%
results.to_pickle("results_classified.pkl")
