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


class PrepareData:
    """
    - Convert CF data into SWDE format
    - Create the labels
    - Remove some data
    -
    """

    def __init__(
        self, featurizer, label_handler, parallel, remove_folder_flag, shortcut, raw_data_folder
    ):
        self.featurizer = Featurizer()
        self.label_handler = LabelHandler()
        self.parallel = parallel
        self.remove_folder_flag = remove_folder_flag
        self.shortcut = shortcut
        self.raw_data_folder = raw_data_folder

    def fit(self):
        if self.remove_folder_flag:
            self.remove_folder(self.raw_data_folder)

        #! Shortcut for debugging:
        if not self.shortcut:
            df = self.load_data(wae_data_load_path, limit=page_limit)  # develop size = 75824
            df = self.label_handler.format_annotation(df)
            df_positives_negatives = self.label_handler.create_postive_negative_data(
                df, negative_fraction=negative_fraction
            )
            df_positives_negatives.to_pickle(f"{dataset_name}_df_positives_negatives_temp.pkl")
        else:
            logging.info(f"Short cutting...")
            df_positives_negatives = pd.read_pickle(
                f"{dataset_name}_df_positives_negatives_temp.pkl"
            )

        logging.info(f"Size of the dataset: {len(df_positives_negatives)}")

        df_domains = list(df_positives_negatives.groupby("domain"))

        # websites_to_debug = ['walkerhamill.com']
        # df_domains = [x for x in df_domains if x[0] in websites_to_debug]

        logging.info(f"Number of Domains: {len(df_domains)}")

        def prepare(domain_name_and_df):
            domain_name, domain_df = domain_name_and_df

            save_folder = self.raw_data_folder / "processed"
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / f"{domain_name}.pkl"

            dedup_save_folder = self.raw_data_folder / "processed_dedup"
            dedup_save_folder.mkdir(parents=True, exist_ok=True)
            dedup_save_path = dedup_save_folder / f"{domain_name}.pkl"

            # if (
            #     not save_path.exists()
            #     and not dedup_save_path.exists()
            #     and domain_name not in ["ciphr.com"]
            # ):
            logging.info(f"domain_name: {domain_name}")

            domain_df["html"] = domain_df.apply(
                lambda row: self.insert_url_into_html(row["url"], row["html"]), axis=1
            )
            domain_df["nodes"] = domain_df["html"].apply(self.featurizer.get_nodes)
            #! During the inference this can be a problem - check how to deal with it when productionalizing
            domain_df = domain_df.dropna(subset=["nodes"])

            #! Training
            domain_df = self.label_handler.add_gt_counts_and_sort(domain_df)
            domain_df = self.label_handler.add_page_id(domain_df)

            self.save_ground_truth(domain_df, domain_name, self.raw_data_folder)
            self.save_htmls(domain_df, domain_name, self.raw_data_folder)

            domain_df["page_features"] = domain_df["nodes"].apply(self.featurizer.get_swde_features)

            domain_df = self.label_handler.add_classification_label_to_nodes(domain_df)

            #! Inference
            domain_df["swde_features"] = domain_df.apply(
                lambda page: self.featurizer.get_swde_features(page["nodes"]), axis=1
            )

            #! Save data
            logging.debug(f"Saved full file at: {save_path} ({len(domain_df)})")
            domain_df.to_pickle(save_path)

            domain_df = self.create_dedup_data(domain_df)
            domain_df["swde_features"] = domain_df.apply(
                lambda page: featurizer.get_swde_features(page["nodes"]), axis=1
            )
            logging.debug(f"Saved dedup file at: {save_path} ({len(domain_df)})")
            domain_df.to_pickle(dedup_save_path)

            return domain_name

        if parallel:
            num_cores = mp.cpu_count()
            with mp.Pool(num_cores) as pool, tqdm(
                total=len(df_domains), desc="Processing data"
            ) as t:
                for res in pool.imap_unordered(prepare, df_domains):
                    # for res in pool.imap(prepare, df_domains):
                    t.set_description(f"Processed {res}")
                    t.update()
        else:
            for df_domain in df_domains:
                prepare(df_domain)

    def predict(self):
        pass

    def load_data(self, load_data_path: str, limit: int = -1) -> pd.DataFrame:
        logging.info("Load_data...")
        self.wae_data_load_path = load_data_path
        self.df = pdx.read_avro(str(load_data_path / "dataset.avro"))
        self.df = self.df[:limit]
        self.df.annotations = self.df.annotations.apply(literal_eval)

        for column in ["url", "domain", "annotations"]:
            assert column in self.df.columns, f"Column: {column} not in DF"

        logging.info(len(self.df))

    def save_ground_truth(self, df, domain_name, root_folder):
        """
        In domain folder save a single csv file with its pages annotations
        """
        logging.debug("Save_ground_truth...")

        folder_path = root_folder / "ground_truth"
        folder_path.mkdir(parents=True, exist_ok=True)

        page_annotations_df = df[["page_id", f"{self.tag}-gt_text_count", f"{self.tag}-gt_text"]]
        page_annotations_df.to_csv(
            folder_path / f"{domain_name}-{self.tag}.csv", sep="\t", index=False
        )

    def save_htmls(self, df: pd.DataFrame, domain_name: str, root_folder: Path):
        """
        In domain folder save all html pages
        """

        def save_html(html, save_path):
            Html_file = open(save_path, "w")
            Html_file.write(html)
            Html_file.close()

        logging.debug("Save htmls...")
        folder_path = root_folder / "htmls" / domain_name
        folder_path.mkdir(parents=True, exist_ok=True)
        df.apply(lambda row: save_html(row["html"], folder_path / f"{row['page_id']}.htm"), axis=1)

    # def save_domain_node_features(self, df, raw_data_folder, domain_name):
    #     folder_path = raw_data_folder / "prepared"
    #     folder_path.mkdir(parents=True, exist_ok=True)
    #     domain_nodes = []
    #     for page_nodes in df["nodes"]:
    #         domain_nodes.extend(page_nodes)
    #     domain_nodes_df = pd.DataFrame(
    #         domain_nodes, columns=["xpath", "text", "node_gt_tag", "node_gt_text"]
    #     )
    #     save_path = folder_path / f"{domain_name}.pkl"
    #     logging.info(f"save_path: {save_path}")
    #     domain_nodes_df.to_pickle(save_path)
    #     return domain_nodes_df

    def remove_folder(self, raw_data_folder: str):
        logging.info("Remove folder...")
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

    def create_dedup_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
        ddd = (
            df_nodes_dedup[["url", "xpath", "node_text", "node_gt_tag", "node_gt_text"]]
            .groupby("url")
            .agg(lambda x: list(x))
            .reset_index()
        )
        # ? Combine dedup nodes into a single column (as they were before )
        ddd["nodes"] = ddd.apply(
            lambda x: list(zip(x["xpath"], x["node_text"], x["node_gt_tag"], x["node_gt_text"])),
            axis=1,
        )
        # ? Remove pages that don't contain nodes anymore due to dedup
        ddd = ddd.dropna(subset=["nodes"])
        # ? Remove columns that are not relevant anymore
        ddd = ddd.drop(["xpath", "node_text", "node_gt_tag", "node_gt_text"], axis=1)
        # ? Merge both datasets (size of ddd might be different from the original df)
        ddd = ddd.set_index("url")
        df = df.set_index("url")
        df = df.drop(["nodes"], axis=1)
        df = ddd.join(df)
        assert len(df) > 0
        return df

    def save_dedup(
        self,
        df: pd.DataFrame,
        domain_name: str,
        raw_data_folder: Path,
    ):
        folder_path = raw_data_folder / "dedup"
        folder_path.mkdir(parents=True, exist_ok=True)
        save_path = folder_path / f"{domain_name}.pkl"
        logging.debug(f"Saved dedup file at: {save_path} ({len(df)})")
        df.to_pickle(save_path)


if __name__ == "__main__":
    # wandb.login()
    # self.run = wandb.init(project="LanguageModel", resume=False, tags=["convert_data"])

    FORMAT = "[ %(asctime)s ] %(filename)s:%(lineno)5s - %(funcName)35s() : %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    remove_folder_flag = True
    shortcut = False
    dataset_name = "develop"
    negative_fraction = 0.10  # ? 0.10
    page_limit = -1  # ? -1
    parallel = True

    # ? Full version
    wae_data_load_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}")
    # ? Smaller version
    # wae_data_load_path = Path(f"/data/GIT/delete/")

    raw_data_folder = Path(f"/data/GIT/delete-/{dataset_name}")

    featurizer = Featurizer()
    label_handler = LabelHandler()

    preparer = PrepareData(featurizer=featurizer, label_handler=label_handler, parallel=parallel, remove_folder_flag=remove_folder_flag, shortcut=shortcut, raw_data_folder=raw_data_folder)
    preparer.load_data(wae_data_load_path)

    preparer.fit()

    # self.run.save[""]()
    # self.run.finish()
