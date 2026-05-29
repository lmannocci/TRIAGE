from utils.log_manager.log_manager import *

from joblib import dump, load
import pickle
# import uunet.multinet as ml
import pandas as pd
from utils.decorator_definition import *

# absolute_path = os.path.dirname(__file__)
# results = os.path.join(absolute_path, f"..{os.sep}..{os.sep}results{os.sep}")
file_name = os.path.splitext(os.path.basename(__file__))[0]


class Checkpoint:
    def __init__(self):
        self.lm = LogManager('main')

    # def __get_path(self, filename, dir_path, add_prefix):
    #     if dir_path == None:
    #         if add_prefix == True:
    #             path = results + self.filename + '_' + filename
    #         else:
    #             path = results + filename
    #     else:
    #         if add_prefix == True:
    #             path = dir_path + self.filename + '_' + filename
    #         else:
    #             path = dir_path + filename
    #     return path

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    #dir_path = None, add_prefix = True

    def save_object(self, obj, path):
        #path = self.__get_path(filename, dir_path, add_prefix)
        with open(path, 'wb') as f:  # Overwrites any existing file.
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        self.lm.printl(f"saved object: {path}")

    def read_object(self, path):
        # if dir_path == None:
        #     path = results + filename
        # else:
        #     path = dir_path + filename

        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.lm.printl(f"read_object: {path}")
        return obj

    def read_dataframe(self, path, dtype, line_terminator=None, on_bad_lines=None):
        kwargs = {"dtype": dtype}

        if line_terminator is not None:
            kwargs["lineterminator"] = line_terminator
        if on_bad_lines is not None:
            kwargs["on_bad_lines"] = on_bad_lines
        
        df = pd.read_csv(path, **kwargs)
 
        self.lm.printl(f"read_dataframe: {path}")
        return df

    # def read_dataframe(self, path, dtype, line_terminator=None, on_bad_lines=None):
    #     if on_bad_lines is None
    #         if line_terminator is None:
    #             df = pd.read_csv(path, dtype=dtype)
    #         else:
    #             df = pd.read_csv(path, dtype=dtype, lineterminator=line_terminator)
    #     else:
    #         if line_terminator is None:
    #             df = pd.read_csv(path, dtype=dtype, on_bad_lines=on_bad_lines)
    #         else:
    #             df = pd.read_csv(path, dtype=dtype, lineterminator=line_terminator, on_bad_lines=on_bad_lines)
    #     self.lm.printl(f"read_dataframe: {path}")
    #     return df

    def save_dataframe(self, df, path):
        # path = self.__get_path(filename, dir_path, add_prefix)
        df.to_csv(path, index=False)
        self.lm.printl(f"save_dataframe: {path}")

    # def update_dataframe(self, df, path, dtype):
    #     self.lm.printl(f"New dataframe shape: {str(df.shape[0])}")
    #     # Check if the file exists
    #     if os.path.exists(path):
    #         # If the file exists, read it
    #         existing_df = pd.read_csv(path, dtype=dtype)
    #         self.lm.printl(f"Existing dataframe shape: {str(existing_df.shape[0])}")
    #         # Append the new results
    #         updated_df = pd.concat([existing_df, df], ignore_index=True)
    #     else:
    #         self.lm.printl(f"Existing dataframe shape: 0 (first time).")
    #         # If the file does not exist, the updated dataframe is just the result
    #         updated_df = df
    #     self.lm.printl(f"Dataframe to write shape: {str(updated_df.shape[0])}")
    #     updated_df.to_csv(path, index=False)
    #     self.lm.printl(f"update_dataframe: {path}")

    def update_dataframe(self, new_row, path):
        """
        Append a new row to a CSV file as a DataFrame.
        
        Parameters:
        - new_row (dict): A dictionary representing the row to be added.
        - csv_file (str): Path to the CSV file.
        """
        df = pd.DataFrame([new_row])  # Convert dictionary to DataFrame
        file_exists = os.path.exists(path)  # Check if the file exists

        df.to_csv(path, mode='a', header=not file_exists, index=False)
        # self.lm.printl(f"update_dataframe: {path}")

    def update_csv_inplace(self, df: pd.DataFrame, path: str) -> None:
        """
        Safely overwrite a CSV file with a modified DataFrame.

        Steps:
        1. Save to a temporary file with suffix '_mod'
        2. Atomically replace the original file

        Args:
            df   : pandas DataFrame to save
            path : path to existing CSV file
        """

        # Create temporary path
        if path.endswith(".csv"):
            mod_path = path.replace(".csv", "_mod.csv")
        else:
            raise ValueError("Only CSV files are supported")

        # Save updated dataframe
        df.to_csv(mod_path, index=False)

        # Replace original file atomically
        os.replace(mod_path, path)

        self.lm.printl(f"update_csv_inplace: {path}")

    
    def update_columns_dataframe(self, df, path, join_columns, dtype):
        """
            Reads a dataframe from the given file path, performs an inner join with another dataframe,
            and saves the resulting dataframe to a specified output path.

            Parameters:
            - path (str): Path to the input CSV file to read the dataframe from and to write to.
            - df (pd.DataFrame): The new dataframe to join with.
            - join_columns (list or str): Columns to use for the inner join.

            Returns:
            - pd.DataFrame: The resulting dataframe after the join.
            """
        # Read the existing dataframe from the file
        input_df = pd.read_csv(path, dtype=dtype)

        # Perform the inner join, updating the existing dataframe
        result_df = input_df.merge(df, on=join_columns, how='inner')

        # Save the resulting dataframe
        result_df.to_csv(path, index=False)

        return result_df

    # def read_multiplex_network(self, path):
    #     MG = ml.read(file=path)
    #     self.lm.printl(f"read_multilayer_network: {path}")
    #     return MG

    # def save_multiplex_network(self, MG, path):
    #     ml.write(MG, file=path)
    #     self.lm.printl(f"save_multilayer_network: {path}")

    def save_model(self, model, path):
        # Save the model
        dump(model, path)
        self.lm.printl(f"saved model: {path}")

    def load_model(self, path):
        # Load the model
        loaded_model = load(path)
        self.lm.printl(f"loaded model: {path}")
        return loaded_model

    def save_txt(self, s, path):
        with open(path, "w") as f:
            f.write(s)
        self.lm.printl(f"save_txt: {path}")

    def read_txt(self, path):
        with open(path, "r") as f:
            s = f.read(path)
        self.lm.printl(f"read_txt: {path}")
        return s
        