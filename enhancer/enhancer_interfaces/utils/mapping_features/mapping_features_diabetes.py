# ------------------------------------------------------------------
    # Manual alias mapping: alias -> canonical feature or list of features
    # Add / extend entries here when needed.
    # ------------------------------------------------------------------
feature_alias_map = {
    # HighBP
    "high blood pressure": "HighBP",
    "hypertension": "HighBP",

    # HighChol
    "high cholesterol": "HighChol",
    "cholesterol levels": "HighChol",
    "cholesterol level": "HighChol",

    # CholCheck
    "cholesterol check": "CholCheck",
    "cholesterol screening": "CholCheck",
    "cholesterol check history": "CholCheck",
    "cholesterol checked": "CholCheck",
    "cholesterol checked in the last 5 years": "CholCheck",
    "cholesterol levels checked": "CholCheck",
    "has had cholesterol levels checked in the last 5 years": "CholCheck",
    "cholesterol check in the last 5 years": "CholCheck",

    # BMI
    "bmi": "BMI",
    "body mass index": "BMI",
    "obesity": "BMI",
    "overweight": "BMI",
    "overweight status": "BMI",
    "obesity history": "BMI",
    "unhealthy body weight": "BMI",
    "normal weight": "BMI",
    "healthy weight": "BMI",
    "weight": "BMI",

    # Smoker
    "smoking": "Smoker",
    "smoking history": "Smoker",
    "history of smoking": "Smoker",
    "smoker": "Smoker",
    "no smoking": "Smoker",
    "no smoking history": "Smoker",
    "no smoking habit": "Smoker",
    "absence of smoking": "Smoker",
    "absence of smoking history": "Smoker",
    "lack of smoking": "Smoker",
    "lack of smoking history": "Smoker",
    "smoked at least 100 cigarettes in its entire life": "Smoker",

    # Stroke
    "stroke": "Stroke",
    "history of stroke": "Stroke",

    # HeartDiseaseorAttack
    "coronary heart disease": "HeartDiseaseorAttack",
    "myocardial infarction": "HeartDiseaseorAttack",
    "heart disease or myocardial infarction": "HeartDiseaseorAttack",
    "coronary heart disease or myocardial infarction": "HeartDiseaseorAttack",
    "coronary heart disease/myocardial infarction": "HeartDiseaseorAttack",
    "history of coronary heart disease or myocardial infarction": "HeartDiseaseorAttack",
    "absence of coronary heart disease or myocardial infarction": "HeartDiseaseorAttack",
    "absence of coronary heart disease/myocardial infarction": "HeartDiseaseorAttack",
    "no coronary heart disease or myocardial infarction": "HeartDiseaseorAttack",
    "no coronary heart disease/myocardial infarction": "HeartDiseaseorAttack",

    # PhysActivity
    "physical activity": "PhysActivity",
    "physical activity level": "PhysActivity",
    "physical activity levels": "PhysActivity",
    "physical activity in the last 30 days": "PhysActivity",
    "physical activity in last 30 days": "PhysActivity",
    "physical activity habits": "PhysActivity",
    "lack of physical activity in the last 30 days": "PhysActivity",

    # Fruits
    "fruit consumption": "Fruits",
    "fruits": "Fruits",
    "consumes fruits": "Fruits",
    "fruit intake": "Fruits",

    # Veggies
    "vegetable consumption": "Veggies",
    "vegetables": "Veggies",
    "vegetables consumption": "Veggies",
    "consumes vegetables regularly": "Veggies",
    "consumption of vegetables": "Veggies",
    "consumption of vegetables daily": "Veggies",
    "vegetables consumed regularly": "Veggies",
    "vegetables consumed 1 or more times per day": "Veggies",
    "regular consumption of vegetables": "Veggies",
    "lack of vegetable consumption": "Veggies",
    "absence of vegetable consumption": "Veggies",
    "lack of regular vegetable consumption": "Veggies",

    # HvyAlcoholConsump
    "heavy alcohol consumption": "HvyAlcoholConsump",
    "alcohol consumption": "HvyAlcoholConsump",
    "high alcohol intake": "HvyAlcoholConsump",

    # AnyHealthcare
    "health insurance": "AnyHealthcare",
    "health insurance status": "AnyHealthcare",
    "any kind of health care coverage, including health insurance": "AnyHealthcare",
    "access to care": "AnyHealthcare",
    "access to healthcare": "AnyHealthcare",
    "access to healthcare resources": "AnyHealthcare",
    "access to medical care": "AnyHealthcare",
    "healthcare access": "AnyHealthcare",
    "lack of healthcare access": "AnyHealthcare",
    "lack of health insurance": "AnyHealthcare",
    "limited access to healthcare resources": "AnyHealthcare",

    # NoDocbcCost
    "cost barrier to doctor visit": "NoDocbcCost",
    "difficulty seeing a doctor due to cost": "NoDocbcCost",
    "ability to see a doctor due to cost": "NoDocbcCost",
    "ability to afford medical care": "NoDocbcCost",
    "access to medical care due to cost": "NoDocbcCost",
    "inability to see a doctor due to cost": "NoDocbcCost",
    "difficulty affording medical care": "NoDocbcCost",
    "difficulty affording medical care in the past": "NoDocbcCost",
    "difficulty accessing care due to cost": "NoDocbcCost",
    "difficulty in accessing medical care": "NoDocbcCost",
    "no doctor visit due to cost": "NoDocbcCost",
    "no doctor visit due to cost in the past 12 months": "NoDocbcCost",
    "no doctor visit in past 12 months due to cost": "NoDocbcCost",
    "no doctor visit in past 12 months": "NoDocbcCost",
    "no doc due to cost": "NoDocbcCost",
    "need to see a doctor but could not due to cost": "NoDocbcCost",
    "was there a time in the past 12 months when it needed to see a doctor but could not because of cost": "NoDocbcCost",
    "history of not being able to see a doctor due to cost": "NoDocbcCost",
    "history of not seeing a doctor when needed due to cost": "NoDocbcCost",
    "no recent doctor visits due to cost": "NoDocbcCost",

    # GenHlth
    "general health": "GenHlth",
    "good general health": "GenHlth",
    "good general health rating": "GenHlth",
    "overall health": "GenHlth",
    "overall health rating": "GenHlth",
    "overall health score": "GenHlth",
    "overall health status": "GenHlth",
    "health status": "GenHlth",
    "low general health rating": "GenHlth",
    "low general health score": "GenHlth",
    "low overall health rating": "GenHlth",

    # MentHlth
    "mental health": "MentHlth",
    "mental health days": "MentHlth",
    "poor mental health": "MentHlth",
    "poor mental health days": "MentHlth",
    "mental health in the past 30 days": "MentHlth",
    "mental health in the last 30 days": "MentHlth",
    "mental health condition": "MentHlth",
    "mental health condition in the past 30 days": "MentHlth",
    "mental health condition over the past 30 days": "MentHlth",
    "mental health conditions": "MentHlth",
    "mental health issues": "MentHlth",
    "mental health status": "MentHlth",
    "mental health status in the past 30 days": "MentHlth",
    "mental health score": "MentHlth",
    "number of poor mental health days": "MentHlth",
    "number of days with poor mental health": "MentHlth",
    "number of days in which mental health condition was not good in the past 30 days": "MentHlth",
    "no mental health issues in the last 30 days": "MentHlth",
    "no mental health conditions": "MentHlth",
    "low mental health": "MentHlth",
    "depression": "MentHlth",

    # PhysHlth
    "physical health": "PhysHlth",
    "physical health days": "PhysHlth",
    "poor physical health": "PhysHlth",
    "poor physical health days": "PhysHlth",
    "physical health in the past 30 days": "PhysHlth",
    "physical health in the last 30 days": "PhysHlth",
    "physical health condition": "PhysHlth",
    "physical health condition in the past 30 days": "PhysHlth",
    "physical health condition over the past 30 days": "PhysHlth",
    "physical health conditions": "PhysHlth",
    "physical health issues": "PhysHlth",
    "physical health status": "PhysHlth",
    "physical health status in the past 30 days": "PhysHlth",
    "physical health score": "PhysHlth",
    "physical health not good": "PhysHlth",
    "physical health problems in the past 30 days": "PhysHlth",
    "physical health problems in the last 30 days": "PhysHlth",
    "number of poor physical health days": "PhysHlth",
    "number of days with poor physical health": "PhysHlth",
    "number of physical health days in the past 30 days": "PhysHlth",
    "number of days in which physical health condition was not good in the past 30 days": "PhysHlth",
    "no days in which physical health condition was not good in the past 30 days": "PhysHlth",

    # DiffWalk
    "difficulty walking": "DiffWalk",
    "difficulty in walking": "DiffWalk",
    "history of difficulty walking": "DiffWalk",
    "history of difficulty in walking": "DiffWalk",
    "difficulty walking": "DiffWalk",
    "mobility": "DiffWalk",
    "mobility issue": "DiffWalk",
    "mobility issues": "DiffWalk",
    "good mobility": "DiffWalk",
    "no difficulty walking": "DiffWalk",
    "no difficulty in walking": "DiffWalk",
    "absence of difficulty walking": "DiffWalk",
    "absence of difficulty in walking": "DiffWalk",
    "does not have difficulty in walking": "DiffWalk",

    # Sex
    "sex": "Sex",
    "gender": "Sex",
    "male": "Sex",
    "female": "Sex",
    "male gender": "Sex",
    "female gender": "Sex",
    "is male": "Sex",
    "is female": "Sex",

    # Age
    "age": "Age",

    # Education
    "education": "Education",
    "education level": "Education",
    "level of education": "Education",

    # Income
    "income": "Income",
    "income level": "Income",
    "socioeconomic status": "Income",
}