import pandas as pd
from task1_unplanned_readmission import build_index_cohort_with_labels

admissions = pd.read_csv("admissions_demo.csv")   # 1 wiersz na stay_id (stays/oddziały)
diagnoses  = pd.read_csv("diagnoses_demo.csv")    # 1 wiersz na kod ICD-10 powiązany ze stay_id
labeled = build_index_cohort_with_labels(admissions, diagnoses)
labeled.to_csv("index_cohort_with_labels.csv", index=False)
