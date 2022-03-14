# ---
# jupyter:
#   jupytext:
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
from markuplmft.fine_tuning.run_swde.lm_model import LModel
lm = LModel()
lm.load_data()
lm.prepare_model_to_train()
lm.fit()

# %%
print("Done Training!")
