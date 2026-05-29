import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from utils.checkpoint.checkpoint import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from typing import Optional
from utils.common_variables import *

absolute_path = os.path.dirname(__file__)
# file_name = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")

class Preprocessing:
    def __init__(self, ch: Checkpoint, lm: LogManager, dataset_prefix: str):
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm
        self.icm: IntegrityConstraintManager = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}.csv"

        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

    @staticmethod
    def __preprocessing_pima(df: pd.DataFrame) -> pd.DataFrame:
        df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[
            ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

        df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
        df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
        df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())
        df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
        df['BMI'] = df['BMI'].fillna(df['BMI'].median())
        df[ind] = df.index  # Adding an index column

        return df

    @staticmethod
    def __preprocessing_diabetes(df: pd.DataFrame) -> pd.DataFrame:
        mapping_diabetes_columns = {'Diabetes_binary': 'Outcome'}
        df = df.rename(columns=mapping_diabetes_columns)

        df[ind] = df.index  # Adding an index column
        df['Outcome'] = df.astype({'Outcome': 'int64'})['Outcome']  # Ensure Outcome is of type int64
        return df


    def __preprocessing_stroke(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop the ID column as it's not informative
        df.drop(columns=['id'], inplace=True)
        # Standardize column names
        df.rename(columns={'Residence_type': 'residence_type', 'stroke': 'Outcome'}, inplace=True)

        # Segment by gender and age bin for better BMI imputation
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 120], 
                                labels=['child', 'young_adult', 'adult', 'senior', 'elderly'])

        # Function to impute BMI per gender + age_group segment
        def impute_bmi(row):
            if pd.notnull(row['bmi']):
                return row['bmi']
            median_bmi = df[(df['gender'] == row['gender']) & 
                            (df['age_group'] == row['age_group'])]['bmi'].median()
            if np.isnan(median_bmi):  # fallback if segment is too small
                return df['bmi'].median()
            return median_bmi

        df['bmi'] = df.apply(impute_bmi, axis=1)

        # Remove helper column
        df.drop(columns=['age_group'], inplace=True)

        # Encode categorical columns using LabelEncoder
        label_encoders = {}
        for col in self.info['cat_columns']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure all are strings to avoid NaNs
            label_encoders[col] = le  # Save encoder if you want to inverse later

        # Save LabelEncoders
        self.ch.save_object(label_encoders, f"{data_path}{self.dataset_prefix}_label_encoders.pkl")
        

        df[ind] = df.index  # Adding an index column
        df['Outcome'] = df.astype({'Outcome': 'int64'})['Outcome']  # Ensure Outcome is of type int64
        return df

    def __preprocessing_liver(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rename columns to camelCase
        df.rename(columns={
            "Age of the patient": "age",
            "Gender of the patient": "gender",
            "Total Bilirubin": "totalBilirubin",
            "Direct Bilirubin": "directBilirubin",
            "Alkphos Alkaline Phosphotase": "alkalinePhosphatase",
            "Sgpt Alamine Aminotransferase": "alanineAminotransferase",
            "Sgot Aspartate Aminotransferase": "aspartateAminotransferase",
            "Total Protiens": "totalProteins",
            "ALB Albumin": "albumin",
            "A/G Ratio Albumin and Globulin Ratio": "albuminGlobulinRatio",
            "Result": "Outcome"
        }, inplace=True)

        # Missing values count
        # age                            2
        # gender                       902
        # totalBilirubin               648
        # directBilirubin              561
        # alkalinePhosphatase          796
        # alanineAminotransferase      538
        # aspartateAminotransferase    462
        # totalProteins                463
        # albumin                      494
        # albuminGlobulinRatio         559
        # liverDisease                   0

        # Drop rows with more than 50% missing values
        row_threshold = int(df.shape[1] * 0.5)
        df = df[df.isnull().sum(axis=1) <= row_threshold]

        # Impute 'gender' with mode based on Outcome
        df["gender"] = df.groupby("Outcome")["gender"].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "Male")
        )

        cat_columns = ['gender']
        num_columns = [
            "age", "totalBilirubin", "directBilirubin", "alkalinePhosphatase",
            "alanineAminotransferase", "aspartateAminotransferase",
            "totalProteins", "albumin", "albuminGlobulinRatio"
        ]

        # Fill NaNs in numerical columns with median based on gender + Outcome
        for col in num_columns:
            df[col] = df.groupby(["gender", "Outcome"])[col].transform(
                lambda x: x.fillna(x.median())
            )

        
        # Optional: map gender to binary
        # df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
        # Encode categorical columns using LabelEncoder
        label_encoders = {}
        for col in self.info['cat_columns']:
            self.lm.printl(f"Encoding column: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure all are strings to avoid NaNs
            self.lm.printl(le.classes_)
            self.lm.printl(le.transform(le.classes_))
            label_encoders[col] = le  # Save encoder if you want to inverse later

        # Save LabelEncoders
        self.ch.save_object(label_encoders, f"{data_path}{self.dataset_prefix}_label_encoders.pkl")

        # Optional: map target (assuming 1 = liver disease, 2 = no disease or vice versa)
        df["Outcome"] = (df["Outcome"] == 1).astype(int)  # adjust as needed

        df[ind] = df.index  # Adding an index column
        df['Outcome'] = df.astype({'Outcome': 'int64'})['Outcome']
        return df

    def __preprocessing_covid(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove all atributes with less than 5% of present data
        df.drop_duplicates(inplace=True)
        for col_name, col in df.items():
            if df[col_name].isna().sum()/df.shape[0] > 0.95:
                df.drop([col_name], axis=1, inplace=True)

        # Transform categorical classes into numeric classes
        # Consider missing data on categorical exams as negatives, since the med staff considered not necessary
        bin_attributes = ['Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1',
                        'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3',
                        'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E',
                        'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2', 'Strepto A']

        for col_name in bin_attributes:
            df[col_name].fillna('negative', inplace=True)

        df.replace(to_replace='not_detected', value=int(0), inplace=True)
        df.replace(to_replace='detected', value=int(1), inplace=True)
        df.replace(to_replace='negative', value=int(0), inplace=True)
        df.replace(to_replace='positive', value=int(1), inplace=True)

        # Remove unecessary or duplicated atributes
        df.drop(['Patient ID', 'Influenza A, rapid test', 'Influenza B, rapid test'], axis=1, inplace=True)

        # Transform the 3 columns of Patient Admition into one (if later we decide to also predict this atribute)
        patient_admition =  df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] * 2 + \
                            df['Patient addmited to intensive care unit (1=yes, 0=no)'] *3 + \
                            df['Patient addmited to regular ward (1=yes, 0=no)']
                
        df.insert(3, 'Patient admition', patient_admition)
        df.drop(['Patient addmited to semi-intensive unit (1=yes, 0=no)', 
                'Patient addmited to intensive care unit (1=yes, 0=no)', 
                'Patient addmited to regular ward (1=yes, 0=no)'], 
                axis=1, inplace=True)

        ### Trying Two Diferent Strategies for Data Manipulation
        # Considering the density distribution, ploted above, we can see some particularly diferent distributions between positive and negative cases. That way, we decided to try two diferent approachs to the problem:

        # - cut all the rows that lack more than 80% of data - what has left with fewer cases and a bad positive/negative distribution, but less cases with none data;
        # - cut all the rows that lack more than 80% of data, and are not positive cases - what has left us with almost 50/50 distribution, but required more data filling.

        # The first one will be called df_small, and the second and largest one df from now on in the code.

        df_small = df.copy()
        df_large = df.copy()

        for row_index, row in df_small.iterrows():
            if row.isna().sum()/df.shape[1] > 0.20:
                df_small.drop([row_index], inplace=True)
        df_small.reset_index(inplace=True, drop=True)

        for row_index, row in df_large.iterrows():
            if row.isna().sum()/df.shape[1] > 0.20 and row['SARS-Cov-2 exam result'] == 0:
                df_large.drop([row_index], inplace=True)
        df_large.reset_index(inplace=True, drop=True)

        # To fill the ramaining NaN entries, we decided to use the mean value, 
        # since all the atributes with NaN cases are float dtypes.

        # But, to maintain the distribution profile of P and N 
        # cases accros the atributes, we used the positive cases mean for each positive row in every column,
        # and the same for negative cases. Mekr even more evidently the diferences.

        df_small.loc[df['SARS-Cov-2 exam result'] == 1] = df_small[df_small['SARS-Cov-2 exam result'] == 1].fillna(df_small[df_small['SARS-Cov-2 exam result'] == 1].mean())
        df_small.loc[df['SARS-Cov-2 exam result'] == 0] = df_small[df_small['SARS-Cov-2 exam result'] == 0].fillna(df_small[df_small['SARS-Cov-2 exam result'] == 0].mean())

        df_large.loc[df_large['SARS-Cov-2 exam result'] == 1] = df_large[df_large['SARS-Cov-2 exam result'] == 1].fillna(df_large[df_large['SARS-Cov-2 exam result'] == 1].mean())
        df_large.loc[df_large['SARS-Cov-2 exam result'] == 0] = df_large[df_large['SARS-Cov-2 exam result'] == 0].fillna(df_large[df_large['SARS-Cov-2 exam result'] == 0].mean())

        pd. set_option('display.max_columns', None)

        # Selecting which dataframe to use
        df = df_large
        
        renamed_columns = {
            'Patient age quantile': 'age_quantile',
            'SARS-Cov-2 exam result': 'Outcome',
            'Patient admition': 'admission',

            'Hematocrit': 'hematocrit',
            'Hemoglobin': 'hemoglobin',
            'Platelets': 'platelets',
            'Mean platelet volume ': 'mean_platelet_volume',
            'Red blood Cells': 'red_blood_cells',
            'Lymphocytes': 'lymphocytes',
            'Mean corpuscular hemoglobin concentration\xa0(MCHC)': 'mchc',
            'Leukocytes': 'leukocytes',
            'Basophils': 'basophils',
            'Mean corpuscular hemoglobin (MCH)': 'mch',
            'Eosinophils': 'eosinophils',
            'Mean corpuscular volume (MCV)': 'mcv',
            'Monocytes': 'monocytes',
            'Red blood cell distribution width (RDW)': 'rdw',

            'Respiratory Syncytial Virus': 'respiratory_syncytial_virus',
            'Influenza A': 'influenza_a',
            'Influenza B': 'influenza_b',
            'Parainfluenza 1': 'parainfluenza_1',
            'CoronavirusNL63': 'coronavirus_nl63',
            'Rhinovirus/Enterovirus': 'rhinovirus_enterovirus',
            'Coronavirus HKU1': 'coronavirus_hku1',
            'Parainfluenza 3': 'parainfluenza_3',
            'Chlamydophila pneumoniae': 'chlamydophila_pneumoniae',
            'Adenovirus': 'adenovirus',
            'Parainfluenza 4': 'parainfluenza_4',
            'Coronavirus229E': 'coronavirus_229e',
            'CoronavirusOC43': 'coronavirus_oc43',
            'Inf A H1N1 2009': 'influenza_a_h1n1_2009',
            'Bordetella pertussis': 'bordetella_pertussis',
            'Metapneumovirus': 'metapneumovirus',
            'Parainfluenza 2': 'parainfluenza_2',

            'Neutrophils': 'neutrophils',
            'Urea': 'urea',
            'Proteina C reativa mg/dL': 'c_reactive_protein_mg_dl',
            'Creatinine': 'creatinine',
            'Potassium': 'potassium',
            'Sodium': 'sodium',
            'Strepto A': 'strepto_a'
        }
        df = df.rename(columns = renamed_columns)

        # I don't use admission for the prediction task
        y_admission = df.pop('admission')
        df['strepto_a'] = df['strepto_a'].astype('int64')

        # Optional: map target (assuming 1 = liver disease, 2 = no disease or vice versa)
        df["Outcome"] = (df["Outcome"] == 1).astype(int)  # adjust as needed

        df[ind] = df.index  # Adding an index column
        df['Outcome'] = df.astype({'Outcome': 'int64'})['Outcome']
        return df

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def preprocess_dataframe(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """

        :param df:
        :return:
        """
        if self.dataset_prefix == 'pima':
            if df is None:
                df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)
            df = self.__preprocessing_pima(df)
        elif self.dataset_prefix == 'diabetes':
            if df is None:
                df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)
            df = self.__preprocessing_diabetes(df)
        elif self.dataset_prefix == 'stroke':
            if df is None:
                df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)
            df = self.__preprocessing_stroke(df)
        elif self.dataset_prefix == 'liver':
            if df is None:
                df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)
            df = self.__preprocessing_liver(df)
        elif self.dataset_prefix == 'covid':
            if df is None:
                df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)
            df = self.__preprocessing_covid(df)
        self.ch.save_dataframe(df, f"{data_path}{self.dataset_prefix}_preprocessed.csv")
        return df

    
    @log_method
    def info_dataframe(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is None:
            df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset_prefix}_preprocessed.csv", dtype=dtype)
        self.lm.printl(f"Dataset: {self.dataset}")
        self.lm.printl(f"Shape: {df.shape}")
        # self.lm.printl(f"Columns: {df.columns.tolist()}")
        # self.lm.printl(f"Data types:\n{df.dtypes}")
        # self.lm.printl(f"Missing values:\n{df.isna().sum()}")
        self.lm.printl(f"Outcome distribution:\n{df['Outcome'].value_counts(normalize=True).round(3)}")
        self.lm.printl("------------------------------------------------------------------")