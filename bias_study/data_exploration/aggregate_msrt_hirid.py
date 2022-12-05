import os

import numpy as np
import pandas as pd

COMMON_PATH = (
    "/cluster/work/grlab/clinical/hirid_public/benchmark/pipeline_runs/run_tls/"
)
OUTPUT_PATH = (
    "/cluster/work/grlab/projects/projects2022-icu-biases/preprocessed_data/HiRID"
)
MERGED_PATH = os.path.join(COMMON_PATH, "merged_stage")

VASOP_INOP_VAR = [f"pm{i}" for i in range(39, 47)]


def get_stats_var(df, old_var, new_var):
    return pd.concat(
        [
            df.mean().rename({old_var: f"mean_{new_var}"}, axis=1),
            df.median().rename({old_var: f"median_{new_var}"}, axis=1),
            df.quantile(0.25).rename({old_var: f"q1_{new_var}"}, axis=1),
            df.quantile(0.75).rename({old_var: f"q3_{new_var}"}, axis=1),
            df.min().rename({old_var: f"min_{new_var}"}, axis=1),
            df.max().rename({old_var: f"max_{new_var}"}, axis=1),
        ],
        axis=1,
    )


def flag_has_pharma(df, group, var):
    start = None
    for j in group[var].dropna().index:
        val = group.loc[j, var]
        if val > 0 and start is None:
            start = j
        if val == 0 and start is not None:
            df.loc[start:j, "has_vasop_inop"] = True
            start = None
    return df


def agg_var_conditioned_pharma(df):
    for patientid, group in df.groupby('patientid'):
        for var in VASOP_INOP_VAR:
            df = flag_has_pharma(df, group, var)
    df_w_pharma = df[df["has_vasop_inop"]]
    df_wo_pharma = df[~df["has_vasop_inop"]]
    df_res = pd.DataFrame()
    for var in ["vm1", "vm5"]:
        df_res = pd.concat(
            [
                df_res,
                get_stats_var(df_w_pharma[['patientid', var]].dropna().groupby('patientid'), var, f"{var}_with_vasop_inop"),
            ], axis=1
        )
        df_res = pd.concat(
            [
                df_res,
                get_stats_var(
                    df_wo_pharma[['patientid', var]].dropna().groupby('patientid'), var, f"{var}_without_vasop_inop"
                ),
            ], axis=1
        )
    return df_res


def classic_agg_var(df):
    df_res = pd.DataFrame()
    for var in ["vm1", "vm5", "vm136", "vm146"]:
        df_res = pd.concat([df_res, get_stats_var(df[['patientid', var]].dropna().groupby('patientid'), var, var)], axis=1)
    return df_res


def agg_msrt(parquet_file):
    df = pd.read_parquet(parquet_file)
    df["has_vasop_inop"] = False
    df_stats_msrt = classic_agg_var(df)
    df_stats_msrt_cond = agg_var_conditioned_pharma(df)
    return pd.concat([df_stats_msrt, df_stats_msrt_cond], axis=1)


def main():
    agg_msrt_patients = pd.DataFrame()
    for filename in os.listdir(MERGED_PATH):
        if os.path.splitext(filename)[1] == ".parquet":
            agg_msrt_patients = pd.concat(
                [agg_msrt_patients, agg_msrt(os.path.join(MERGED_PATH, filename))]
            )

    agg_msrt_patients.to_csv(os.path.join(OUTPUT_PATH, "med_msrt_stats_patients.csv"))
