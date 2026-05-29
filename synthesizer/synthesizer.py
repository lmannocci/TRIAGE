import pandas as pd
import os
import numpy as np

from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager

from typing import List, Tuple, Dict, Union, Optional
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdmetrics.single_table import NewRowSynthesis
import torch


absolute_path = os.path.dirname(__file__)
# file_name = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class Synthesizer:
    def __init__(self, ch: Checkpoint, lm: LogManager, dataset_prefix: str, synthesizer_name: str, epochs: int = 300, n_samples_synthesizer: int = 1000, cuda_visible_devices: str = "1,2,3"):
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)
        self.icm.check_synthesizer(synthesizer_name)
        self.icm.check_synthesizer_epochs(epochs)

        self.dm: DirectoryManager = DirectoryManager(lm, results_path, dataset_prefix=dataset_prefix, synthesizer_name=synthesizer_name, epochs=epochs, n_samples_synthesizer=n_samples_synthesizer)

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}_preprocessed.csv"
        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

        self.synthesizer_name: str = synthesizer_name
        self.epochs: int = epochs
        self.n_samples_synthesizer: int = n_samples_synthesizer
        self.cuda_visible_devices: str = cuda_visible_devices
        self.__compute_metadata()

        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        # Verify GPU selection
        self.lm.printl(f"Number of GPUs selected: {str(torch.cuda.device_count())}")
    
        self.cuda_devices = (
            [int(x) for x in self.cuda_visible_devices.split(",")]
            if self.cuda_visible_devices else [0]
        )

        self.lm.printl(f"CUDA available: {torch.cuda.is_available()}")


    @log_method
    def __get_train_data(self, df: pd.DataFrame):
        train_idx_path = f"{self.dm.dataset_path}train_idx.csv"
        test_idx_path = f"{self.dm.dataset_path}test_idx.csv"

        train_idx = self.ch.read_dataframe(train_idx_path, dtype=dtype)[ind]
        test_idx = self.ch.read_dataframe(test_idx_path, dtype=dtype)[ind]

        # Define train and test sets
        train_df = df[~df[ind].isin(test_idx)].copy()
        test_df = df[df[ind].isin(test_idx)].copy()

        return train_df, test_df
    
    @log_method
    def __compute_metadata(self):
        df = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)

        if os.path.exists(f"{self.dm.synthesizer_path}synthesizer_metadata.json"):
            self.metadata = Metadata.load_from_json(filepath=f"{self.dm.synthesizer_path}synthesizer_metadata.json")
            self.lm.printl(f"Metadata loaded from: {self.dm.synthesizer_path}synthesizer_metadata.json")
        else:
            self.metadata = Metadata.detect_from_dataframe(data=df, table_name=self.dataset_prefix)
            self.metadata.save_to_json(filepath=f"{self.dm.synthesizer_path}synthesizer_metadata.json")
            self.lm.printl(f"Metadata computed and saved at: {self.dm.synthesizer_path}synthesizer_metadata.json")
    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    


    @log_method
    def run_synthesizer(self):
        if os.path.exists(f"{self.dm.synthesizer_path}synthesizer.pkl"):
            self.lm.printl(f"Synthesizer already exists at: {self.dm.synthesizer_path}synthesizer.pkl, skipping training.")
            return
        
        if self.metadata is None:
            self.compute_metadata()
        df = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype) 
        train_df, test_df = self.__get_train_data(df)
        if self.synthesizer_name == "ctgan":
            synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs = self.epochs,
                verbose=True
            )

        # Fit the synthesizer to the training data
        synthesizer.fit(train_df)

        synthesizer.save(f"{self.dm.synthesizer_path}synthesizer.pkl")
        
        self.lm.printl(f"Synthesizer saved at: {self.dm.synthesizer_path}synthesizer.pkl")
    
    @log_method
    def generate_synthetic_data(self) -> pd.DataFrame:
        synthesizer = CTGANSynthesizer.load(f"{self.dm.synthesizer_path}synthesizer.pkl")
        self.lm.printl(f"Synthesizer loaded from: {self.dm.synthesizer_path}synthesizer.pkl")
        
        synthetic_df = synthesizer.sample(self.n_samples_synthesizer)
        synthetic_df["original_index"] = range(len(synthetic_df)) # Overwrite original_index with new sequential index for synthetic data
        self.lm.printl(f"Generated {self.n_samples_synthesizer} synthetic samples.")
        
        self.ch.save_dataframe(synthetic_df, f"{self.dm.synthesizer_path}{self.dataset_prefix}_synthetic.csv")
    
    @log_method
    def evaluate_synthetic_resemblance(self):
        """
        Compute SDV diagnostic + quality resemblance between the original data and the
        synthetic data, save:
        1. a local one-row dataframe in self.dm.synthesizer_path
        2. a global aggregated dataframe in self.dm.global_evaluator_path

        The global dataframe is keyed by:
            - dataset_prefix
            - synthesizer_name
            - epochs
            - n_samples_synthesizer

        If a row with the same keys already exists, it is overwritten (keep='last').
        """

        # ---------------------------------------------------------
        # Load data
        # ---------------------------------------------------------
        original_df = self.ch.read_dataframe(
            f"{data_path}{self.dataset}",
            dtype=dtype
        )
        synthetic_df = self.ch.read_dataframe(
            f"{self.dm.synthesizer_path}{self.dataset_prefix}_synthetic.csv",
            dtype=dtype
        )

        # Evaluate on test set only (avoid leakage from train to synthetic)
        train_original_df, test_original_df = self.__get_train_data(original_df)

        # ---------------------------------------------------------
        # SDV evaluation
        # ---------------------------------------------------------
        diagnostic = run_diagnostic(
            real_data=test_original_df,
            synthetic_data=synthetic_df,
            metadata=self.metadata
        )

        quality = evaluate_quality(
            real_data=test_original_df,
            synthetic_data=synthetic_df,
            metadata=self.metadata
        )

        diagnostic_score = float(diagnostic.get_score())
        quality_score = float(quality.get_score())
        quality_props = quality.get_properties()

        # ---------------------------------------------------------
        # Novelty / memorization check (TRAIN vs SYNTH)
        # ---------------------------------------------------------
        try:
            table_name = self.dataset_prefix
            single_table_meta = self.metadata.to_dict()["tables"][table_name]

            novelty_score = float(
                NewRowSynthesis.compute(
                    real_data=train_original_df,
                    synthetic_data=synthetic_df,
                    metadata=single_table_meta
                )
            )
            novelty_err = None
        except Exception as e:
            novelty_score = np.nan
            novelty_err = repr(e)

        # ---------------------------------------------------------
        # Print results
        # ---------------------------------------------------------
        self.lm.printl("=== SDV Single-Table Evaluation (CTGAN) ===")
        self.lm.printl(f"Diagnostic score (test vs synth): {diagnostic_score:.4f}")
        self.lm.printl(f"Quality score    (test vs synth): {quality_score:.4f}")

        if isinstance(quality_props, pd.DataFrame) and not quality_props.empty:
            self.lm.printl("Quality properties breakdown:")
            self.lm.printl(quality_props.to_string(index=False))

        self.lm.printl("=== Novelty / Memorization Check ===")
        if not np.isnan(novelty_score):
            self.lm.printl(
                f"NewRowSynthesis score (train vs synth): {novelty_score:.4f}  (higher = more novel)"
            )
        else:
            self.lm.printl("NewRowSynthesis score: n/a (failed to compute)")
            self.lm.printl(f"Reason: {novelty_err}")

        # ---------------------------------------------------------
        # Build one-row metrics dataframe
        # ---------------------------------------------------------
        metrics = {
            "dataset_prefix": self.dataset_prefix,
            "synthesizer_name": self.synthesizer_name,
            "epochs": self.epochs,
            "n_samples_synthesizer": self.n_samples_synthesizer,

            "diagnostic_score": diagnostic_score,
            "quality_score": quality_score,
            "new_row_synthesis_score": novelty_score,

            "n_rows_original": int(len(original_df)),
            "n_rows_original_train": int(len(train_original_df)),
            "n_rows_original_test": int(len(test_original_df)),
            "n_rows_synthetic": int(len(synthetic_df)),
            "n_cols": int(test_original_df.shape[1]),
        }

        # Flatten SDV quality properties into columns
        if isinstance(quality_props, pd.DataFrame) and not quality_props.empty:
            for _, r in quality_props.iterrows():
                prop_name = str(r["Property"]).strip().lower().replace(" ", "_")
                metrics[f"quality_{prop_name}_score"] = float(r["Score"])

        metrics_df = pd.DataFrame([metrics])

        # ---------------------------------------------------------
        # Save local metrics dataframe
        # ---------------------------------------------------------
        local_path = f"{self.dm.synthesizer_path}resemblance_metrics.csv"
        self.ch.save_dataframe(metrics_df, local_path)
        
        # ---------------------------------------------------------
        # Save / update global metrics dataframe
        # ---------------------------------------------------------
        global_path = f"{self.dm.global_evaluator_path}synthesizer_resemblance_metrics.csv"

        key_cols = [
            "dataset_prefix",
            "synthesizer_name",
            "epochs",
            "n_samples_synthesizer",
        ]

        if os.path.exists(global_path):
            global_df = self.ch.read_dataframe(global_path, dtype=dtype)
            global_df = pd.concat([global_df, metrics_df], ignore_index=True)
        else:
            global_df = metrics_df.copy()

        global_df = global_df.drop_duplicates(subset=key_cols, keep="last")

        self.ch.save_dataframe(global_df, global_path)