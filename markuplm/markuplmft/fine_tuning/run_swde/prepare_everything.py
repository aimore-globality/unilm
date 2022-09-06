import logging
import pandas as pd
from tqdm import tqdm
import pandavro as pdx
from ast import literal_eval
from pathlib import Path
from lxml import html as lxml_html
import wandb
import shutil
import multiprocessing as mp
from pathlib import Path
from pathlib import Path
from markuplmft.fine_tuning.run_swde.featurizer import Featurizer
from markuplmft.fine_tuning.run_swde.labelhandler import LabelHandler
import os
import glob
from typing import Tuple


class PrepareData:
    """
    - For each page creates an HTML into a folder separated by domain 
    - Creates a .csv file with all the labels per page. One .csv file per domain. 
    - Creates the node_dataset .
    """

    def __init__(self, parallel:bool, remove_folder_flag:bool, shortcut:bool, raw_data_folder):
        self.parallel = parallel
        self.remove_folder_flag = remove_folder_flag
        self.shortcut = shortcut
        self.raw_data_folder = raw_data_folder

    def load_data(self, load_data_path: str, limit: int = -1) -> pd.DataFrame:
        logging.info("Load_data...")
        df = pdx.read_avro(str(load_data_path / "dataset.avro"))
        df = df[:limit]
        df.annotations = df.annotations.apply(literal_eval)  # TODO: Change this for the new data

        for column in ["url", "domain", "annotations"]:
            assert column in df.columns, f"Column: {column} not in DF"

        logging.info(len(df))
        return df

    def save_ground_truth(self, df, domain_name:str, root_folder:Path):
        """
        In domain folder save a single csv file with its pages annotations
        """
        logging.debug("Save_ground_truth...")

        folder_path = root_folder / "ground_truth"
        folder_path.mkdir(parents=True, exist_ok=True)

        page_annotations_df = df[
            [
                "page_id",
                f"{label_handler.tag}-gt_text_count",
                f"{label_handler.tag}-gt_text",
            ]
        ]
        page_annotations_df.to_csv(
            folder_path / f"{domain_name}-{label_handler.tag}.csv", sep="\t", index=False
        )

    def save_htmls(self, df, domain_name: str, root_folder: Path):
        """
        Create all html pages for the domain in the domain folder 
        """

        def save_html(html, save_path):
            Html_file = open(save_path, "w")
            Html_file.write(html)
            Html_file.close()

        logging.debug("Save htmls...")
        folder_path = root_folder / "htmls" / domain_name
        folder_path.mkdir(parents=True, exist_ok=True)
        df.apply(lambda row: save_html(row["html"], folder_path / f"{row['page_id']}.htm"), axis=1)

    def remove_folder(self, raw_data_folder: str):
        """
        Remove the dataset folder to start fresh
        """
        self.raw_data_folder = raw_data_folder
        if os.path.exists(self.raw_data_folder):
            logging.info(f"Overwriting this folder: \n{self.raw_data_folder}")
            try:
                shutil.rmtree(self.raw_data_folder)
                logging.info(f"REMOVED: {self.raw_data_folder}")
            except OSError as e:
                logging.info(f"Error: {e.filename} - {e.strerror}.")

        groundtruth_folder_path = self.raw_data_folder
        groundtruth_folder_path.mkdir(parents=True, exist_ok=True)

    def create_dedup_data(self, df) -> pd.DataFrame:
        """
        Randomly remove duplicated nodes across the entire domain
        Note: Some urls might have been droped due to the deduplication
        """
        # ? Expand nodes
        df_nodes = df.explode("nodes").reset_index()
        # ? Join expanded nodes into df
        df_nodes = df_nodes.join(
            pd.DataFrame(
                df_nodes.pop("nodes").tolist(),
                columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
            )
        )

        # ? Remove duplicates
        df_nodes_dedup = df_nodes.drop_duplicates("node_text")
        # ? Group dedup nodes by page
        df_dedup = (
            df_nodes_dedup[["url", "xpath", "node_text", "node_gt_tag", "node_gt_text"]]
            .groupby("url")
            .agg(lambda x: list(x))
            .reset_index()
        )
        # ? Combine dedup nodes into a single column (as they were before)
        df_dedup["nodes"] = df_dedup.apply(
            lambda row: list(
                zip(row["xpath"], row["node_text"], row["node_gt_tag"], row["node_gt_text"])
            ),
            axis=1,
        )
        # ? Remove pages that don't contain nodes anymore due to dedup
        df_dedup = df_dedup.dropna(subset=["nodes"])
        # ? Remove columns that are not relevant anymore
        df_dedup = df_dedup.drop(["xpath", "node_text", "node_gt_tag", "node_gt_text"], axis=1)
        # ? Merge both datasets (size of df_dedup might be different from the original df)
        df_dedup = df_dedup.set_index("url")
        df = df.set_index("url")
        df = df.drop(["nodes"], axis=1)
        df = df_dedup.join(df)
        assert len(df) > 0
        df.reset_index(inplace=True)
        return df


def save_processed_data(df, save_path: Path):
    save_path.parents[0].mkdir(parents=True, exist_ok=True)
    logging.info(f"Saved data ({len(df)}) at: {save_path}")
    df.to_pickle(save_path)


def prepare_domain(domain_name_and_domain_df) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    domain_name, domain_df = domain_name_and_domain_df
    logging.info(f"domain_name: {domain_name}")

    #! Inference
    domain_df["html"] = domain_df.apply(
        lambda row: featurizer.insert_url_into_html(row["url"], row["html"]), axis=1
    )

    domain_df["nodes"] = domain_df["html"].apply(featurizer.get_nodes)

    #! During the inference not being able to convert a html to nodes can be a problem - check how to deal with it when productionalizing
    original_numb_pages = len(domain_df)
    domain_df = domain_df.dropna(subset=["nodes"])
    final_numb_pages = len(domain_df)
    logging.info(f"Number of pages before: {original_numb_pages} | after: {final_numb_pages} ({100*final_numb_pages/original_numb_pages:.1f} %)")

    #! Training
    domain_df = label_handler.add_gt_counts_and_sort(domain_df)
    domain_df = label_handler.add_page_id(domain_df)
    if domain_name in debug_domain_names:
        print("FOUND")
    domain_df = label_handler.add_classification_label_to_nodes(domain_df)

    preparer.save_ground_truth(domain_df, domain_name, raw_data_folder)
    preparer.save_htmls(domain_df, domain_name, raw_data_folder)

    domain_df_dedup = preparer.create_dedup_data(domain_df)

    return domain_name, domain_df, domain_df_dedup


if __name__ == "__main__":

    wandb.login()
    run = wandb.init(project="LanguageModel", resume=False, tags=["convert_data"])

    FORMAT = "[ %(asctime)s ] %(filename)25s:%(lineno)5s - %(funcName)35s() : %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    remove_folder_flag = False
    shortcut = False
    # negative_fraction = 0.1  # ? 0.10
    negative_fraction = 1
    parallel = True
    with_img = True
    root_folder_path = Path(f"/data/GIT/prepared_data")

    debug_flag = False
    debug_domain_names = ['1820productions.com', 'wc.com']
    
    if with_img:
        with_img_str = "with_imgs"
    else:
        with_img_str = "without_imgs"

    # neg_str = f"neg_{str(negative_fraction)}"
    neg_str = f"neg_1"

    data_folder_name = f"node_classifier_{with_img_str}_{neg_str}"

    if debug_flag:
        data_folder_name = data_folder_name + "_DEBUG"
        parallel = False

    logging.info(f"name_root_folder: {data_folder_name}")
    featurizer = Featurizer()
    label_handler = LabelHandler()

    for dataset_name in ["develop", "train"][:1]:
        logging.info(f"DATASET: {dataset_name}")

        wae_data_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}")
        logging.info(f"Loaded data from: {wae_data_path}")

        raw_data_folder = root_folder_path / data_folder_name / dataset_name

        save_path = raw_data_folder / "processed.pkl"
        dedup_save_path = raw_data_folder / "processed_dedup.pkl"
        if not save_path.exists() and not dedup_save_path.exists() or remove_folder_flag:
            #! Prepare data for training #TODO: Move into a function
            preparer = PrepareData(
                parallel=parallel,
                remove_folder_flag=remove_folder_flag,
                shortcut=shortcut,
                raw_data_folder=raw_data_folder,
            )

            wae_df = preparer.load_data(wae_data_path)

            #! Shortcut for debugging:
            if not preparer.shortcut:
                df = label_handler.format_annotation(wae_df)
                df = label_handler.remove_non_html_pages(df)
                df_positives_negatives = label_handler.create_postive_negative_data(
                    df,
                    negative_fraction=negative_fraction,
                    wae_data_path=wae_data_path,
                    with_img=with_img,
                )
                df_positives_negatives.to_pickle(f"{dataset_name}_df_positives_negatives_temp.pkl")
            else:
                logging.info(f"Short cutting...")
                df_positives_negatives = pd.read_pickle(
                    f"{dataset_name}_df_positives_negatives_temp.pkl"
                )

            logging.info(f"# of pages in the dataset: {len(df_positives_negatives)}")

            domains_name_df = list(df_positives_negatives.groupby("domain")) #? Tuple (domain_url, domain_df)

            if debug_flag:
                domains_name_df = [name_df for name_df in domains_name_df if name_df[0] in debug_domain_names]

            logging.info(f"Number of Domains: {len(domains_name_df)}")
            # TODO: Simplify this function - Be careful when running in all data with parallelize - struct.error: 'I' format requires 0 <= number <= 4294967295

            all_df, all_df_dedup = pd.DataFrame(), pd.DataFrame()
            if parallel:
                num_cores = mp.cpu_count()
                with mp.Pool(num_cores) as pool, tqdm(
                    total=len(domains_name_df), desc="Preparing domain data"
                ) as iteration:
                    for domain_name, domain_df, domain_df_dedup in pool.imap_unordered(
                        prepare_domain, domains_name_df
                    ):
                        all_df = all_df.append(domain_df)
                        all_df_dedup = all_df_dedup.append(domain_df_dedup)
                        iteration.set_description(f"Processed {domain_name}")
                        iteration.update()
            else:
                for df_domain in domains_name_df:
                    domain_name, domain_df, domain_df_dedup = prepare_domain(df_domain)
                    all_df = all_df.append(domain_df)
                    all_df_dedup = all_df_dedup.append(domain_df_dedup)

            all_df = all_df.sort_values(['domain', 'url'])
            save_processed_data(all_df, save_path)

            all_df_dedup = all_df_dedup.sort_values(['domain', 'url'])
            save_processed_data(all_df_dedup, dedup_save_path)
        else:
            load_path = str(raw_data_folder / "processed_dedup.pkl")
            save_path = Path(load_path.replace(".pkl", "_feat.pkl"))
            
            #! Genereate Features #TODO: Move into a function
            dfs = pd.read_pickle(load_path)

            num_data_points = len(dfs)
            print(num_data_points)

            def return_page_features(url_nodes):
                url, nodes = url_nodes
                return featurizer.get_page_features(url, nodes)

            if parallel:
                num_cores = mp.cpu_count()
                with mp.Pool(num_cores) as pool, tqdm(
                    total=num_data_points, desc="Processing data"
                ) as iteration:
                    all_page_features = []
                    # for res in pool.imap_unordered(cache_page_features, all_df.values):
                    for page_features in pool.imap(
                        return_page_features, dfs[["url", "nodes"]].values
                    ):
                        all_page_features.append(page_features)
                        iteration.update()
                    print(len(all_page_features))
                    dfs["page_features"] = all_page_features
            else:
                dfs.apply(
                    lambda page: featurizer.get_page_features(page["url"], page["nodes"]),
                    axis=1,
                )

            save_processed_data(dfs, save_path)

    run.save()
    run.finish()