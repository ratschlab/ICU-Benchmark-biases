import os

import pandas as pd

COMMON_PATH = (
    "/cluster/work/grlab/clinical/hirid_public/benchmark/pipeline_runs/run_tls/"
)
PATIENT_PATH = os.path.join(COMMON_PATH, "general_table_extended.parquet")
MERGED_PATH = os.path.join(COMMON_PATH, "merged_stage")


def extract_length_stay(info_patients):
    info_patients["lengthstay"] = info_patients["endtime"] - info_patients["starttime"]
    return info_patients


def extract_date_stay(df):
    sort_date_patients = (
        df[df["vm1"].notna()]
        .sort_values("datetime")
        .groupby("patientid")[["patientid", "datetime"]]
    )
    start_date_patients = (
        sort_date_patients.head(1)
        .rename({"datetime": "starttime"}, axis=1)
        .set_index("patientid")
    )
    end_date_patients = (
        sort_date_patients.tail(1)
        .rename({"datetime": "endtime"}, axis=1)
        .set_index("patientid")
    )
    return start_date_patients.join(end_date_patients)


def extract_info(parquet_file):
    df = pd.read_parquet(parquet_file)
    date_stay_patients = extract_date_stay(df)
    return date_stay_patients


def filter_multiple_index(info_patients):
    start = (
        info_patients.sort_values("starttime")
        .groupby("patientid")[["starttime"]]
        .head(1)
    )
    end = info_patients.sort_values("endtime").groupby("patientid")[["endtime"]].tail(1)
    return start.join(end)


def main():
    patients = pd.read_parquet(PATIENT_PATH).set_index("patientid")
    info_patients = pd.DataFrame()
    for filename in os.listdir(MERGED_PATH):
        if os.path.splitext(filename)[1] == ".parquet":
            info_patients = pd.concat(
                [info_patients, extract_info(os.path.join(MERGED_PATH, filename))]
            )
    if not info_patients.index.is_unique:
        info_patients = filter_multiple_index(info_patients)
    return patients.join(info_patients)
