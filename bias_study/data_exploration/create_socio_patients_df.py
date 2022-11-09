import datetime
import os

import pandas as pd

COMMON_PATH = "/cluster/work/grlab/clinical/mimic/MIMIC-III/cdb_1.4/source_data"
OUTPUT_PATH = "/custer/work/grlab/projects/projects2022-icu-biases/preprocessed_data/MIMIC"

PUBLIC_INSURANCE = ["Medicare", "Medicaid", "Government"]
UNSPECIFIED_ETHN = [
    "UNKNOWN/NOT SPECIFIED",
    "PATIENT DECLINED TO ANSWER",
    "UNABLE TO OBTAIN",
]
ETHN = ["WHITE", "LATINO", "BLACK", "ASIAN", "MIDDLE EASTERN", "NATIVE AMERICAN"]
OTHER_ETHN = ["OTHER", "MULTI RACE ETHNICITY"]


def map_ethnicity(ethnicity):
    if "WHITE" in ethnicity or "PORTUGUESE" in ethnicity:
        return "WHITE"
    if "BLACK" in ethnicity or "CARIBBEAN" in ethnicity:
        return "BLACK"
    if "LATINO" in ethnicity or "SOUTH AMERICA" in ethnicity:
        return "LATINO"
    if "ASIAN" in ethnicity:
        return "ASIAN"
    if "MIDDLE EASTERN" in ethnicity:
        return "MIDDLE EASTERN"
    if (
        ethnicity == "AMERICAN INDIAN/ALASKA NATIVE"
        or ethnicity == "AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE"
    ):
        return "NATIVE AMERICAN"
    if ethnicity in unspecified_ethn:
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


def read_specific_col_table(table_name, cols=None):
    return pd.read_csv(os.path.join(COMMON_PATH, table_name), usecols=cols)


def create_age_admission(df_admissions, df_patients):
    df_admissions = pd.merge(df_admissions, df_patients["DOB"], on="SUBJECT_ID")
    df_admissions["ADMITDATE"] = df_admissions["ADMITTIME"].apply(
        lambda d: datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").date()
    )
    df_admissions["DOB"] = df_admissions["DOB"].apply(
        lambda d: datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").date()
    )
    df_admissions["AGE"] = (df_admissions["ADMITDATE"] - df_admissions["DOB"]).apply(
        lambda t: t.days
    ) / 365.25


def filter_adult_patients(df_admission):
    return df_admission[df_admission["AGE"] > 16]["SUBJECT_ID"]


def merge_insurance(insurance_type):
    if insurance_type in PUBLIC_INSURANCE:
        return "Public"
    return insurance_type


def create_patient_insurance(df_patients, df_last_admission):
    df_patients["INSURANCE"] = df_last_admission[["SUBJECT_ID", "INSURANCE"]].set_index(
        "SUBJECT_ID"
    )
    df_patients["INSURANCE_3"] = df_patients["INSURANCE"].apply(merge_insurance)
    return df_last_admission


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
    df_patients["LAST_ADMISSION_TYPE"] = df_last_admission_patients[
        ["SUBJECT_ID", "ADMISSION_TYPE"]
    ].set_index("SUBJECT_ID")
    df_patients["EVER_EMERGENCY"] = df_admissions.groupby("SUBJECT_ID").apply(
        lambda df: merge(df, "EMERGENCY")
    )
    df_patients["EVER_URGENT"] = df_admissions.groupby("SUBJECT_ID").apply(
        lambda df: merge(df, "URGENT")
    )
    df_patients["EVER_ELECTIVE"] = df_admissions.groupby("SUBJECT_ID").apply(
        lambda df: merge(df, "ELECTIVE")
    )
    return df_patients


def create_patient_los(df_patients, df_last_admission, df_admissions, df_stays):
    hadm_los = pd.DataFrame(df_stays.groupby("HADM_ID")["LOS"].sum())
    hadm_los["SUBJECT_ID"] = df_admissions[["SUBJECT_ID", "HADM_ID"]].set_index(
        "HADM_ID"
    )
    df_patients["LAST_LOS"] = hadm_los[
        hadm_los.index.isin(df_last_admission["HADM_ID"])
    ].set_index("SUBJECT_ID")["LOS"]
    df_patients["MEAN_LOS"] = hadm_los.groupby("SUBJECT_ID").mean()
    df_patients["NB_ADMISSIONS"] = df_admissions.groupby("SUBJECT_ID").size()
    return df_patients


def main():
    patients = read_specific_col_table(
        "PATIENTS.csv", ["SUBJECT_ID", "GENDER", "DOB"]
    ).set_index("SUBJECT_ID")
    admissions = read_specific_col_table("ADMISSIONS.csv")
    admissions = create_age_admission(admissions, patients)
    stays = read_specific_col_table("ICUSTAYS.csv", ["SUBJECT_ID", "HADM_ID", "LOS"])
    last_admission = (
        admissions.sort_values("ADMITTIME", ascending=False)
        .groupby("SUBJECT_ID")
        .head(1)
    )
    patients = create_patient_insurance(patients, last_admission)
    patients = create_patient_ethnicity(patients, admissions)
    patients = create_patient_language(patients, admissions)
    patients = create_patient_los(patients, last_admission, admissions, stays)
    subject_ids_adult = filter_adult_patients(last_admission)
    patients_adult = patients[subject_ids_adult]
    patients_adult.to_csv(os.path.join(OUTPUT_PATH, 'socioinfo_patients.csv'))
