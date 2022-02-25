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
#
# DONE READ!
#


r"""To pack all the swde html page files into a single pickle file.

This script is to generate a single file to pack up all the content of htmls.
"""

from __future__ import absolute_import, division, print_function

import os
import pickle
import sys
from pathlib import Path

import tqdm
from absl import app, flags

FLAGS = flags.FLAGS

# Flags related to input data.
flags.DEFINE_string("input_swde_path", "", "The root path to swde html page files.")
flags.DEFINE_string("output_pack_path", "", "The file path to save the packed data.")


def pack_swde_data(swde_path, pack_path):
    """Packs the swde dataset to a single file.

    Args:
      swde_path: The path to SWDE dataset pages.
      pack_path: The path to save packed SWDE dataset file.
    Returns:
      None
    """

    swde_path = Path(swde_path)

    swde_data = {}
    print("Loading data...")

    websites_folder = os.listdir(os.path.join(swde_path))

    for website_folder in tqdm.tqdm(websites_folder, desc="Websites - Progress Bar", leave=True):
        html_filenames = os.listdir(os.path.join(swde_path, website_folder))
        html_filenames.sort()
        for html_filename in html_filenames:
            html_file_relative_path = os.path.join(website_folder, html_filename)
            print(f"Page: {html_file_relative_path}")

            html_file_absolute_path = os.path.join(swde_path, html_file_relative_path)
            with open(html_file_absolute_path) as webpage_html:
                html_str = webpage_html.read()

            page = dict(
                website=website_folder,  # E.g. 'capturagroup.com(8)'
                path=html_file_relative_path,  # E.g. 'capturagroup.com(8)/0000.htm'
                html_str=html_str,
            )

            website = website_folder.split("(")[0]
            swde_data[website] = page
    print("Saving data...")
    with open(pack_path, "wb") as output_file:
        pickle.dump(swde_data, output_file)


def main(_):
    pack_swde_data(
        swde_path=FLAGS.input_swde_path,
        pack_path=FLAGS.output_pack_path,
    )


if __name__ == "__main__":
    app.run(main)
