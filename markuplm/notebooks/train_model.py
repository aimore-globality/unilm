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
lm = LModel()
lm.load_data()
lm.prepare_model_to_train()
lm.fit()
