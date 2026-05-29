    def __describe_binary_feature(self, value: int, description: str, feature_name: str, tense: str = "present") -> str:
        if tense not in {"present", "past"}:
            raise ValueError("Tense must be either 'present' or 'past'.")

        verb = "has" if tense == "present" else "had"
        
        if int(value) == 1:
            return f"{verb} {description} ('{feature_name}')"
        elif int(value) == 0:
            return f"{verb} not {description} ('{feature_name}')"
        else:
            raise ValueError("Value must be 0 or 1.")
        
    
    def __create_patient_question(self, row: pd.Series):
        if self.dataset_prefix == 'pima':
            question: str = self.__pima_create_patient_question(row)
        elif self.dataset_prefix == 'diabetes':
            question: str = self.__diabetes_create_patient_question(row)
        elif self.dataset_prefix == 'stroke':
            question: str = self.__stroke_create_patient_question(row)
        elif self.dataset_prefix == 'liver':
            question: str = self.__liver_create_patient_question(row)
        elif self.dataset_prefix == 'covid':
            question: str = self.__covid_create_patient_question(row)
            # self.lm.printl(f"Question for covid patient created: {question}")
        return question

    # START PIMA METHODS
    # ********************
    # Method to create a textual description of each patient
    def __pima_create_patient_description(self, row: pd.Series):
        # description = (f"The patient is {row['Age']} years old, "
        #                f"has had {row['Pregnancies']} pregnancies, "
        #                f"has a glucose level of {row['Glucose']}, "
        #                f"blood pressure of {row['BloodPressure']}, "
        #                f"a skin thickness of {row['SkinThickness']}, "
        #                f"an insulin level of {row['Insulin']}, "
        #                f"a BMI of {row['BMI']}, "
        #                f"a diabetes pedigree function of {row['DiabetesPedigreeFunction']}.")
        description = (f"The patient has Age: {row['Age']}, "
                       f"has had Pregnancies: {row['Pregnancies']}, "
                       f"has Glucose: {row['Glucose']}, "
                       f"has BloodPressure: {row['BloodPressure']}, "
                       f"has SkinThickness: {row['SkinThickness']}, "
                       f"has Insulin: {row['Insulin']}, "
                       f"has BMI: {row['BMI']}, "
                       f"has DiabetesPedigreeFunction: {row['DiabetesPedigreeFunction']}.")
        return description

    def __pima_create_patient_question(self, row: pd.Series) -> str:
        description = self.__pima_create_patient_description(row)
        # question = f"Based on the following information: {description}, does the patient have diabetes?"
        # question = f"""
        # You are a helpful medical expert. Your task is to classify a patient as having diabetes or not based on the provided medical data
        # and using the relevant documents. Based on the following information: {description}, does the patient have diabetes?
        # Please provide a detailed explanation of the reasoning in "step_by_step_thinking" and a clear binary classification in 
        # "classification" where "1" indicates "Diabetes" and "0" indicates "No Diabetes". In addition, include a section with the feature importance ranking and another
        # section with the specific rules that influenced the classification based on each feature. In these two sections, add only relevant exploited features for the decision.
        # Organize your output in JSON format as Dict"step_by_step_thinking": Str(explanation), "classification": Int(0 or 1), 
        # "feature_importance_ranking": List[Str], "rules_applied": Dict[Str, Str]. Your responses will be used for research purposes only, so please
        # provide a definite and clear answer. 
        # """
        question = f"""
            You are a helpful and reliable medical expert. Your task is to classify whether a patient has diabetes based on their medical data and the provided supporting documents.

            Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

            Based on the following information:
            {description}

            Please answer the question: **Does the patient have diabetes?**

            Your response must be structured as a JSON object with the following fields:

            - `"step_by_step_thinking"`: A clear, medically sound explanation of your reasoning process leading to the classification.
            - `"classification"`: An integer value, where `1` means the patient **has diabetes**, and `0` means the patient **does not have diabetes**.
            - `"feature_importance_ranking"`: A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.
            - `"rules_applied"`: A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.

            Only include features that were relevant to the decision. If a feature did not influence your judgment, omit it from the ranking and rules.

            Your answer will be used for research purposes only, so please provide a well-justified and definite response.
        """
        return question


    # Method to extract the classification from answer
    def __pima_extract_classification(self, answer: str) -> Union[int, None]:
        # Usa una regex per cercare 'classification' seguito da ': 0' o ': 1'
        match = re.search(r'classification["\']?\s*:\s*(\d)', answer)
        if match:
            return int(match.group(1))  # Restituisce 0 o 1 come intero
        return None  # Se non trova nulla, ritorna None

    # END PIMA METHODS
    # ********************

    # START DIABETES METHODS
    # ********************
    
    def __diabetes_create_patient_description(self, row: pd.Series):
        # description = (
        #                 f"The patient has Body Mass Index: {row['BMI']}, "
        #                 f"Mental health condition days in past 30 days: {row['MentHlth']}, "
        #                 f"had Physical health condition days in past 30 days: {row['PhysHlth']}, "
        #                 f"has Age: {row['Age']}, "
        #                 f"had High Blood Pressure: {row['HighBP']}, "
        #                 f"had High Cholesterol: {row['HighChol']}, "
        #                 f"has been checked cholestorol levels: {row['CholCheck']}, "
        #                 f"is a Smoker: {row['Smoker']}, "
        #                 f"had a Stroke: {row['Stroke']}, "
        #                 f"had a  Heart Diseaseor Attack: {row['HeartDiseaseorAttack']}, "
        #                 f"engages in Physical Activity: {row['PhysActivity']}, "
        #                 f"consumes Fruits: {row['Fruits']}, "
        #                 f"consumes Veggies: {row['Veggies']}, "
                        
        #                 f"has Heavy Alcohol Consumption: {row['HvyAlcoholConsump']}, "
        #                 f"has access to Healthcare: {row['AnyHealthcare']}, "
        #                 f"needed a doctor but could not because of the cost: {row['NoDocbcCost']}, "
        #                 f"has General Health conditions: {row['GenHlth']}, "
        #                 f"has Difficulty in Walking: {row['DiffWalk']}, "
        #                 f"has Sex: {row['Sex']}, "
        #                 f"on a scale from 1 to 6 has Level of Education: {row['Education']}, "
        #                 f"on a scale from 1 to 8 has Income Level: {row['Income']}.")
        
        description = (
            f"The patient has Body Mass Index: {row['BMI']} ('BMI'), "
            f"number of days in which its mental health condition was not good in the past 30 days: {row['MentHlth']} ('MentHlth'), "
            f"number of days in which its physical health condition was not good in the past 30 days: {row['PhysHlth']} ('PhysHlth'), "
            f"has Age: {row['Age']} ('Age'), "
            f"{self.__describe_binary_feature(row['HighBP'], 'High Blood Pressure', 'HighBP', 'past')}, "
            f"{self.__describe_binary_feature(row['HighChol'], 'High Cholesterol', 'HighChol', 'past')}, "
            f"{self.__describe_binary_feature(row['CholCheck'], 'been checked for cholesterol levels in the last 5 years', 'CholCheck', 'past')}, "
            f"{self.__describe_binary_feature(row['Smoker'], 'smoked at least 100 cigarettes in its entire life', 'Smoker', 'past')}, "
            f"{self.__describe_binary_feature(row['Stroke'], 'a Stroke', 'Stroke', 'past')}, "
            f"{self.__describe_binary_feature(row['HeartDiseaseorAttack'], 'a coronary heart disease or myocardial infarction', 'HeartDiseaseorAttack', 'past')}, "
            f"{self.__describe_binary_feature(row['PhysActivity'], 'engaged in Physical Activity in the last 30 days', 'PhysActivity')}, "
            f"{'consumes Fruits 1 or more times per day' if row['Fruits'] == 1 else 'does not consume Fruits 1 or more times per day'} ('Fruits'), "
            f"{'consumes Vegetables  1 or more times per day' if row['Veggies'] == 1 else 'does not consume Vegetables 1 or more times per day'} ('Veggies'), "
            f"{self.__describe_binary_feature(row['HvyAlcoholConsump'], 'Heavy Alcohol Consumption', 'HvyAlcoholConsump')}, "
            f"{self.__describe_binary_feature(row['AnyHealthcare'], 'any kind of health care coverage, including health insurance', 'AnyHealthcare')}, "
            f"{'was there a time in the past 12 months when it needed to see a doctor but could not because of cost' if int(row['NoDocbcCost']) == 1 else 'was not there a time in the past 12 months when you needed to see a doctor but could not because of cost'} ('NoDocbcCost'), "
            f"on a scale from 1 to 5 in general its health is: {row['GenHlth']} ('GenHlth'), "
            f"{self.__describe_binary_feature(row['DiffWalk'], 'Difficulty in Walking', 'DiffWalk')}, "
            f"{'is male' if int(row['Sex']) == 1 else 'is female'} ('Sex'), "
            f"on a scale from 1 to 6 has Level of Education: {row['Education']} ('Education'), "
            f"on a scale from 1 to 8 has Income Level: {row['Income']} ('Income')."
        )

        return description
    

    
    
    def __diabetes_create_patient_question(self, row: pd.Series) -> str:
        description = self.__diabetes_create_patient_description(row)
        
        # question = f"""
        # You are a helpful medical expert. Your task is to classify a patient as having diabetes or not based on the provided medical data
        # and using the relevant documents. In the following patient's information we specify between parenthesis the features name, which you
        # must use in the output. Based on the following informaiton: {description}, does the patient have diabetes?
        # Please provide a detailed explanation of the reasoning in "step_by_step_thinking" and a clear binary classification in 
        # "classification" where "1" indicates "Diabetes" and "0" indicates "No Diabetes". In addition, include a section with the feature importance ranking and another
        # section with the specific rules that influenced the classification based on each feature. In these two sections, add only relevant exploited features for the decision.
        # Organize your output in JSON format as Dict"step_by_step_thinking": Str(explanation), "classification": Int(0 or 1), 
        # "feature_importance_ranking": List[Str], "rules_applied": Dict[Str, Str]. Your responses will be used for research purposes only, so please
        # provide a definite and clear answer. 
        # """

        question = f"""
            You are a helpful and reliable medical expert. Your task is to classify whether a patient has diabetes based on their medical data and the provided supporting documents.

            Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

            Based on the following information:
            {description}

            Please answer the question: **Does the patient have diabetes?**

            Your response must be structured as a JSON object with the following fields:

            - `"step_by_step_thinking"`: A clear, medically sound explanation of your reasoning process leading to the classification.
            - `"classification"`: An integer value, where `1` means the patient **has diabetes**, and `0` means the patient **does not have diabetes**.
            - `"feature_importance_ranking"`: A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.
            - `"rules_applied"`: A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.

            Only include features that were relevant to the decision. If a feature did not influence your judgment, omit it from the ranking and rules.

            Your answer will be used for research purposes only, so please provide a well-justified and definite response.
        """

        return question
    
    def __diabetes_extract_classification(self, answer: str) -> Union[int, None]:
        # Usa una regex per cercare 'classification' seguito da ': 0' o ': 1'
        match = re.search(r'classification["\']?\s*:\s*(\d)', answer)
        if match:
            return int(match.group(1))  # Restituisce 0 o 1 come intero
        return None  # Se non trova nulla, ritorna None
    # END DIABETES METHODS
    # ********************



    # START STROKE METHODS
    # ********************
    def __stroke_create_patient_description(self, row: pd.Series):
        description = (
            f"The patient "
            f"is {row['gender']} ('gender'), "
            f"has Age: {row['age']} ('age'), "
            f"{self.__describe_binary_feature(row['hypertension'], 'Hypertension', 'hypertension')}, "
            f"{self.__describe_binary_feature(row['heart_disease'], 'Heart Disease', 'heart_disease')}, "
            f"{'is married' if row['ever_married'] == 'Yes' else 'is not married'} ('ever_married'), "
            f"works as {row['work_type']} ('work_type'), "
            f"lives in a {row['residence_type']} area ('Residence_type'), "
            f"has average glucose level: {row['avg_glucose_level']} ('avg_glucose_level'), "
            f"has Body Mass Index: {row['bmi']} ('bmi'), "
            f"{'smoking status: ' + row['smoking_status'] if row['smoking_status'] != 'Unknown' else 'smoking status is unknown'} ('smoking_status'), "
        )

        return description
    

    def __stroke_create_patient_question(self, row: pd.Series) -> str:
        description = self.__stroke_create_patient_description(row)
        question = f"""
            You are a helpful and reliable medical expert. Your task is to classify a patient had a stroke or not based on their medical data and the provided supporting documents.

            Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

            Based on the following information:
            {description}

            Please answer the question: **Did the patient have a stroke?**

            Your response must be structured as a JSON object with the following fields:

            - `"step_by_step_thinking"`: A clear, medically sound explanation of your reasoning process leading to the classification.
            - `"classification"`: An integer value, where `1` means the patient **had a stroke**, and `0` means the patient **did not have a stroke**.
            - `"feature_importance_ranking"`: A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.
            - `"rules_applied"`: A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.

            Only include features that were relevant to the decision. If a feature did not influence your judgment, omit it from the ranking and rules.

            Your answer will be used for research purposes only, so please provide a well-justified and definite response."""
        return question
    
    def __stroke_extract_classification(self, answer: str) -> Union[int, None]:
        # Usa una regex per cercare 'classification' seguito da ': 0' o ': 1'
        match = re.search(r'classification["\']?\s*:\s*(\d)', answer)
        if match:
            return int(match.group(1))  # Restituisce 0 o 1 come intero
        return None  # Se non trova nulla, ritorna None
    # END STROKE METHODS
    # ********************


    # START LIVER METHODS
    # ********************
    def __liver_create_patient_description(self, row: pd.Series):
        description = (f"The patient is {row['age']} years old ('age'), "
                       f"has Gender: {row['gender']} ('gender'), "
                          f"has total bilirubin: {row['totalBilirubin']} ('totalBilirubin'), "
                          f"has direct bilirubin: {row['directBilirubin']} ('directBilirubin'), "
                            f"has alkaline phosphotase: {row['alkalinePhosphatase']} ('alkalinePhosphatase'), "
                            f"has alamine aminotransferase: {row['alanineAminotransferase']} ('alanineAminotransferase'), "
                            f"has aspartate aminotransferase: {row['aspartateAminotransferase']} ('aspartateAminotransferase'), "
                            f"has total protiens: {row['totalProteins']} ('totalProteins'), "
                            f"has albumin: {row['albumin']} ('albumin'), "
                            f"has albumin and globulin ratio: {row['albuminGlobulinRatio']} ('albuminGlobulinRatio').")
        return description
    
    def __liver_create_patient_question(self, row: pd.Series) -> str:
        description = self.__liver_create_patient_description(row)
        question = f"""
            You are a helpful and reliable medical expert. Your task is to classify whether a patient has liver disease based on their medical data and the provided supporting documents.
            Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.
            Based on the following information:
            {description} 
            Please answer the question: **Does the patient have liver disease?**
            Your response must be structured as a JSON object with the following fields:
            - `"step_by_step_thinking"`: A clear, medically sound explanation of your reasoning process leading to the classification.
            - `"classification"`: An integer value, where `1` means the patient **has liver disease**, and `0` means the patient **does not have liver disease**.
            - `"feature_importance_ranking"`: A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.
            - `"rules_applied"`: A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.
            Only include features that were relevant to the decision. If a feature did not influence your judgment, omit it from the ranking and rules.
            Your answer will be used for research purposes only, so please provide a well-justified and definite response.
        """
        return question
    
    def __liver_extract_classification(self, answer: str) -> Union[int, None]:
        # Usa una regex per cercare 'classification' seguito da ': 0' o ': 1'
        match = re.search(r'classification["\']?\s*:\s*(\d)', answer)
        if match:
            return int(match.group(1))  # Restituisce 0 o 1 come intero
        return None  # Se non trova nulla, ritorna None
    
    # END LIVER METHODS
    # ********************


    # START COVID METHODS
    # ********************
    def __covid_create_patient_description(self, row: pd.Series):
        description = (
            f"The patient has age quantile: {row['age_quantile']} ('age_quantile'), "
            f"hematocrit level: {row['hematocrit']} ('hematocrit'), "
            f"hemoglobin level: {row['hemoglobin']} ('hemoglobin'), "
            f"platelet count: {row['platelets']} ('platelets'), "
            f"mean platelet volume: {row['mean_platelet_volume']} ('mean_platelet_volume'), "
            f"red blood cell count: {row['red_blood_cells']} ('red_blood_cells'), "
            f"lymphocyte count: {row['lymphocytes']} ('lymphocytes'), "
            f"MCHC (mean corpuscular hemoglobin concentration): {row['mchc']} ('mchc'), "
            f"leukocyte count: {row['leukocytes']} ('leukocytes'), "
            f"basophil count: {row['basophils']} ('basophils'), "
            f"MCH (mean corpuscular hemoglobin): {row['mch']} ('mch'), "
            f"eosinophil count: {row['eosinophils']} ('eosinophils'), "
            f"MCV (mean corpuscular volume): {row['mcv']} ('mcv'), "
            f"monocyte count: {row['monocytes']} ('monocytes'), "
            f"RDW (red cell distribution width): {row['rdw']} ('rdw'), "
            
            f"{self.__describe_binary_feature(row['respiratory_syncytial_virus'], 'a Respiratory Syncytial Virus infection', 'respiratory_syncytial_virus')}, "
            f"{self.__describe_binary_feature(row['influenza_a'], 'an Influenza A infection', 'influenza_a')}, "
            f"{self.__describe_binary_feature(row['influenza_b'], 'an Influenza B infection', 'influenza_b')}, "
            f"{self.__describe_binary_feature(row['parainfluenza_1'], 'a Parainfluenza 1 infection', 'parainfluenza_1')}, "
            f"{self.__describe_binary_feature(row['coronavirus_nl63'], 'a Coronavirus NL63 infection', 'coronavirus_nl63')}, "
            f"{self.__describe_binary_feature(row['rhinovirus_enterovirus'], 'a Rhinovirus/Enterovirus infection', 'rhinovirus_enterovirus')}, "
            f"{self.__describe_binary_feature(row['coronavirus_hku1'], 'a Coronavirus HKU1 infection', 'coronavirus_hku1')}, "
            f"{self.__describe_binary_feature(row['parainfluenza_3'], 'a Parainfluenza 3 infection', 'parainfluenza_3')}, "
            f"{self.__describe_binary_feature(row['chlamydophila_pneumoniae'], 'a Chlamydophila pneumoniae infection', 'chlamydophila_pneumoniae')}, "
            f"{self.__describe_binary_feature(row['adenovirus'], 'an Adenovirus infection', 'adenovirus')}, "
            f"{self.__describe_binary_feature(row['parainfluenza_4'], 'a Parainfluenza 4 infection', 'parainfluenza_4')}, "
            f"{self.__describe_binary_feature(row['coronavirus_229e'], 'a Coronavirus 229E infection', 'coronavirus_229e')}, "
            f"{self.__describe_binary_feature(row['coronavirus_oc43'], 'a Coronavirus OC43 infection', 'coronavirus_oc43')}, "
            f"{self.__describe_binary_feature(row['influenza_a_h1n1_2009'], 'an Influenza A H1N1 2009 infection', 'influenza_a_h1n1_2009')}, "
            f"{self.__describe_binary_feature(row['bordetella_pertussis'], 'a Bordetella pertussis infection', 'bordetella_pertussis')}, "
            f"{self.__describe_binary_feature(row['metapneumovirus'], 'a Metapneumovirus infection', 'metapneumovirus')}, "
            f"{self.__describe_binary_feature(row['parainfluenza_2'], 'a Parainfluenza 2 infection', 'parainfluenza_2')}, "

            f"neutrophil count: {row['neutrophils']} ('neutrophils'), "
            f"urea level: {row['urea']} ('urea'), "
            f"C-reactive protein level in mg/dL: {row['c_reactive_protein_mg_dl']} ('c_reactive_protein_mg_dl'), "
            f"creatinine level: {row['creatinine']} ('creatinine'), "
            f"potassium level: {row['potassium']} ('potassium'), "
            f"sodium level: {row['sodium']} ('sodium'), "
            f"{self.__describe_binary_feature(row['strepto_a'], 'a Strepto A infection', 'strepto_a')}. "
        )
        return description

    def __covid_create_patient_question(self, row: pd.Series) -> str:
        description = self.__covid_create_patient_description(row)
        question = f"""
            You are a helpful and reliable medical expert. Your task is to classify whether a patient has COVID-19 based on their medical data and the provided supporting documents.
            Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.
            Based on the following information:
            {description} 
            Please answer the question: **Does the patient have COVID-19?**
            Your response must be structured as a JSON object with the following fields:
            - `"step_by_step_thinking"`: A clear, medically sound explanation of your reasoning process leading to the classification.
            - `"classification"`: An integer value, where `1` means the patient **has COVID-19**, and `0` means the patient **does not have COVID-19**.
            - `"feature_importance_ranking"`: A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.
            - `"rules_applied"`: A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.
            Only include features that were relevant to the decision. If a feature did not influence your judgment, omit it from the ranking and rules.
            Your answer will be used for research purposes only, so please provide a well-justified and definite response.
        """
        return question

    def __covid_extract_classification(self, answer: str) -> Union[int, None]:
        match = re.search(r'classification["\']?\s*:\s*(\d)', answer)
        if match:
            return int(match.group(1))  # Restituisce 0 o 1 come intero
        return None  # Se non trova nulla, ritorna None
    # END COVID METHODS
    # ********************

    def __extract_classification(self, answer: str):
        if self.dataset_prefix == 'pima':
            classification: str = self.__pima_extract_classification(answer)
        elif self.dataset_prefix == 'diabetes':
            classification: str = self.__diabetes_extract_classification(answer)
        elif self.dataset_prefix == 'stroke':
            classification: str = self.__stroke_extract_classification(answer)
        elif self.dataset_prefix == 'liver':
            classification: str = self.__liver_extract_classification(answer)
        elif self.dataset_prefix == 'covid':
            # To be implemented
            classification: str = self.__covid_extract_classification(answer)
        return classification



    # Define a function to handle individually missing values for PIMA dataset. Unfeasible for greater datasets, so disabled
    # def __handle_missing_values(self, row):
    #     if row.isnull().any():  # Only modify rows with at least one null value
    #         if row[ind] == 227:
    #             answer_df = self.ch.read_dataframe(f"{self.dm.temp_answer_path}index_{str(int(row[ind]))}_medrag.csv", dtype=dtype)
    #             self.lm.printl(f"Answer for patient 227 still invalid format: {answer_df}.")
    #         if row[ind] in [516, 603, 648, 734]:
    #             eval_answer = ast.literal_eval(row['medrag_answer']) # it is a tuple of one element
    #             # the string has e second part with an explanation that can be discarde. 
    #             #Then the string can be evaulated again and we have the standard dictionary
    #             if row[ind] in [516, 734]:
    #                 s_split = 'Explanation'
    #             elif row[ind] == 603:
    #                 s_split = ' Explanation: The'
    #             elif row[ind] == 648:
    #                 s_split = 'The given'
                    
    #             answer_dict = ast.literal_eval(eval_answer[0].split(s_split)[0]) 
    #             temp_list = []
    #             temp_list.append(answer_dict)
    #             temp_list.append(eval_answer[1])
    #             temp_list.append(eval_answer[2])
    #             temp_tuple = tuple(temp_list)

    #             row['medrag_pred'] = answer_dict['classification']
    #             row['step_by_step_thinking'] = answer_dict['step_by_step_thinking']
    #             row['feature_importance_ranking'] = answer_dict['feature_importance_ranking']
    #             row['rules_applied'] =  answer_dict['rules_applied']
    #             row['medrag_answer'] = str(temp_tuple)
    #     return row  # If no missing values, return unchanged

    # Method to extract the classification from answer
    def extract_classification(self, answer: str) -> Union[int, None]:
        # Usa una regex per cercare 'classification' seguito da ': 0' o ': 1'
        match = re.search(r'classification["\']?\s*:\s*(\d)', answer)
        if match:
            return int(match.group(1))  # Restituisce 0 o 1 come intero
        return None  # Se non trova nulla, ritorna None
