import datetime
import os

import pandas as pd

COMMON_PATH = "/cluster/work/grlab/clinical/mimic/MIMIC-III/cdb_1.4/source_data"
OUTPUT_PATH = "/cluster/work/grlab/projects/projects2022-icu-biases/preprocessed_data/MIMIC"

PUBLIC_INSURANCE = ["Medicare", "Medicaid", "Government"]
UNSPECIFIED_ETHN = [
    "UNKNOWN/NOT SPECIFIED",
    "PATIENT DECLINED TO ANSWER",
    "UNABLE TO OBTAIN",
]


def map_ethnicity(ethnicity):
    if "WHITE" in ethnicity or "PORTUGUESE" in ethnicity:
        return "WHITE"
    if "BLACK" in ethnicity or "CARIBBEAN" in ethnicity:
        return "BLACK"
    if "LATINO" in ethnicity or "SOUTH AMERICA" in ethnicity:
        return "LATINO"
    if "ASIAN" in ethnicity:
        return "ASIAN"
    if ethnicity in UNSPECIFIED_ETHN:
        return "UNK"
    else:
        return "OTHER"


def merge_ethn(list_ethn):
    first_ethn = list_ethn[0]
    for ethn in list_ethn[1:]:
        if ethn == "UNK" or ethn == "OTHER" or ethn == first_ethn:
            continue
        if first_ethn == "UNK" or first_ethn == "OTHER":
            first_ethn = ethn
        else:
            print(list_ethn)
            return "CANT_MERGE"
    return first_ethn


def read_specific_col_table(table_name, cols=None, date_cols=False):
    return pd.read_csv(os.path.join(COMMON_PATH, table_name), usecols=cols, parse_dates=date_cols)


def create_patient_age(df_patients, df_last_admission):
    df_last_admission = pd.merge(df_last_admission, df_patients["DOB"], on="SUBJECT_ID")
    df_last_admission["AGE"] = (df_last_admission["ADMITTIME"].dt.date - df_last_admission["DOB"].dt.date).apply(
        lambda t: t.days
    ) / 365.25
    df_patients['AGE'] = df_last_admission[['SUBJECT_ID', 'AGE']].set_index('SUBJECT_ID')
    return df_patients


def filter_adult_patients(df_patients):
    return df_patients[df_patients["AGE"] > 16]


def merge_insurance(insurance_type):
    if insurance_type in PUBLIC_INSURANCE:
        return "Public"
    return insurance_type


def create_patient_insurance(df_patients, df_last_admission):
    df_patients["INSURANCE"] = df_last_admission[["SUBJECT_ID", "INSURANCE"]].set_index(
        "SUBJECT_ID"
    )
    df_patients["INSURANCE_3"] = df_patients["INSURANCE"].apply(merge_insurance)
    return df_patients


def create_patient_ethnicity(df_patients, df_admissions):
    merged_ethn = {}
    df_admissions["ETHN"] = df_admissions["ETHNICITY"].apply(map_ethnicity)
    patients_more1_adm = df_admissions.groupby("SUBJECT_ID").size()[
        df_admissions.groupby("SUBJECT_ID").size() > 1
    ]
    patients_more1_ethn = (
        df_admissions[df_admissions["SUBJECT_ID"].isin(patients_more1_adm.index)]
        .groupby("SUBJECT_ID")["ETHN"]
        .nunique()[
            df_admissions[df_admissions["SUBJECT_ID"].isin(patients_more1_adm.index)]
            .groupby("SUBJECT_ID")["ETHN"]
            .nunique()
            > 1
        ]
    )

    for name, group in df_admissions[
        df_admissions["SUBJECT_ID"].isin(patients_more1_ethn.index)
    ].groupby("SUBJECT_ID"):
        merged_ethn[name] = merge_ethn(list(group["ETHN"]))
    patients_ethn = {}
    for id in patients.index:
        if id in merged_ethn:
            if merged_ethn[id] != "CANT_MERGE":
                patients_ethn[id] = merged_ethn[id]
            else:
                patients_ethn[id] = None
        else:
            patients_ethn[id] = df_admissions[df_admissions["SUBJECT_ID"] == id][
                "ETHN"
            ].iloc[0]

    df_patients["ETHNICITY"] = pd.Series(patients_ethn)
    return df_patients


def create_patient_language(df_patients, df_admissions):
    df_admissions["SPEAK_ENG"] = df_admissions[df_admissions["LANGUAGE"].notna()][
        "LANGUAGE"
    ].apply(lambda l: l == "ENGL")
    df_patients["SPEAK_ENG"] = (
        df_admissions[df_admissions["LANGUAGE"].notna()]
        .groupby("SUBJECT_ID")["SPEAK_ENG"]
        .sum()
        > 0
    )
    return df_patients


def merge_admission_type(df, key):
    if len(df[df["ADMISSION_TYPE"] == key]):
        return True
    return False


def create_patient_admission_type(df_patients, df_last_admission, df_admissions):
    df_patients["LAST_ADMISSION_TYPE"] = df_last_admission[
        ["SUBJECT_ID", "ADMISSION_TYPE"]
    ].set_index("SUBJECT_ID")
    df_patients["EVER_EMERGENCY"] = df_admissions.groupby("SUBJECT_ID").apply(
        lambda df: merge_admission_type(df, "EMERGENCY")
    )
    df_patients["EVER_URGENT"] = df_admissions.groupby("SUBJECT_ID").apply(
        lambda df: merge_admission_type(df, "URGENT")
    )
    df_patients["EVER_ELECTIVE"] = df_admissions.groupby("SUBJECT_ID").apply(
        lambda df: merge_admission_type(df, "ELECTIVE")
    )
    return df_patients


def create_patient_icu_los(df_patients, df_last_admission, df_admissions, df_stays):
    hadm_los = pd.DataFrame(df_stays.groupby("HADM_ID")["LOS"].sum())
    hadm_los["SUBJECT_ID"] = df_admissions[["SUBJECT_ID", "HADM_ID"]].set_index(
        "HADM_ID"
    )
    df_patients["LAST_ICU_LOS"] = hadm_los[
        hadm_los.index.isin(df_last_admission["HADM_ID"])
    ].set_index("SUBJECT_ID")["LOS"]
    df_patients["MEAN_ICU_LOS"] = hadm_los.groupby("SUBJECT_ID").mean()
    return df_patients

def create_patient_los(df_patients, df_last_admission, df_admissions):
    df_patients["LAST_HOSPITAL_LOS"] = df_last_admission[['SUBJECT_ID', 'HOSPITAL_LOS']].set_index('SUBJECT_ID')
    df_patients["MEAN_HOSPITAL_LOS"] = df_admission.groupby("SUBJECT_ID")['HOSPITAL_LOS'].mean()
    return df_patients

def create_length_admission(df_admissions):
    df_admissions['HOSPITAL_LOS'] = (df_admissions['DISCHTIME'] -df_admissions['ADMITTIME']).dt.days
    return df_admissions

def create_patient_death(df_patients, df_admissions):
    dead_patients = df_admissions[df_admissions['DISCHARGE_LOCATION']=='DEAD/EXPIRED']['SUBJECT_ID']
    death_during_stay = df_admissions[df_admissions['DISCHTIME']>=df_admissions['DEATHTIME']]['SUBJECT_ID']
    df_patients['DEATH_CLOSE_STAY'] = df_patients.index.isin(dead_patients)
    df_patients['DEATH_DURING_STAY'] = df_patients.index.isin(death_during_stay)
    return df_patients


def main():
    patients = read_specific_col_table(
        "PATIENTS.csv", ["SUBJECT_ID", "GENDER", "DOB"], ["DOB"]
    ).set_index("SUBJECT_ID")
    admissions = read_specific_col_table("ADMISSIONS.csv", date_cols=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'])
    stays = read_specific_col_table("ICUSTAYS.csv", ["SUBJECT_ID", "HADM_ID", "LOS"])
    admissions = create_length_admission(admissions)
    last_admission = (
        admissions.sort_values("ADMITTIME", ascending=False)
        .groupby("SUBJECT_ID")
        .head(1)
    )
    patients = create_patient_age(patients, last_admission)
    patients = create_patient_insurance(patients, last_admission)
    patients = create_patient_ethnicity(patients, admissions)
    patients = create_patient_language(patients, admissions)
    patients = create_patient_admission_type(patients, last_admission, admissions)
    patients = create_patient_los(patients, last_admission, admissions)
    patients = create_patient_icu_los(patients, last_admission, admissions, stays)
    patients = create_patient_death(patients, admissions)
    patients_adult = filter_adult_patients(patients)
    patients_adult.to_csv(os.path.join(OUTPUT_PATH, 'socioinfo_patients.csv'))
