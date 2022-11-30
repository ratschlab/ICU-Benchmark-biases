import datetime
import os

import pandas as pd
import numpy as np

COMMON_PATH = (
    "/cluster/work/grlab/clinical/hirid_public/benchmark/pipeline_runs/run_tls/"
)
OUTPUT_PATH = (
    "/cluster/work/grlab/projects/projects2022-icu-biases/preprocessed_data/HiRID"
)
MERGED_PATH = os.path.join(COMMON_PATH, "merged_stage")

BASELINE_INTERVAL = {'vm1': datetime.timedelta(seconds=120), 'vm5': datetime.timedelta(seconds=120), 
'vm136': datetime.timedelta(hours=8)}

def count_missed_msrt(intervals, var):
    nb_strict_missed_msrt = 0
    nb_missed_msrt = 0
    for inter in intervals:
        if inter > BASELINE_INTERVAL[var]:
            nb_strict_missed_msrt += inter // BASELINE_INTERVAL[var]
            nb_missed_msrt += inter // BASELINE_INTERVAL[var] - 1
    return nb_strict_missed_msrt, nb_missed_msrt

def get_interval(datetime_list):
    intervals = []
    for i in range(0, len(datetime_list)-1):
        intervals.append(datetime_list[i+1]-datetime_list[i])
    return intervals

def get_stats_interval(intervals_list, var):
    stats = {}
    stats[f'mean_inter_{var}'] = np.mean(intervals_list) / datetime.timedelta(seconds=1)
    stats[f'median_inter_{var}'] = np.median(intervals_list) / datetime.timedelta(seconds=1)
    stats[f'max_inter_{var}'] = np.max(intervals_list) / datetime.timedelta(seconds=1)
    stats[f'1q_inter_{var}'] = np.quantile(intervals_list, 0.25, interpolation='nearest') / datetime.timedelta(seconds=1)
    stats[f'3q_inter_{var}'] = np.quantile(intervals_list, 0.75, interpolation='nearest') / datetime.timedelta(seconds=1)
    stats[f'nb_strict_missed_{var}'], stats[f'nb_missed_{var}'] = count_missed_msrt(intervals_list, var)
    return stats


def get_stats_interval_per_patient(msr_per_patient, stats_per_patient, var):
    dict_stats_patient = {}
    for patientid, msrts in msr_per_patient:
        intervals = get_interval(list(msrts['datetime']))
        if intervals:
            dict_stats_patient[patientid] = get_stats_interval(intervals, var)
    stats_per_patient = pd.concat([stats_per_patient, pd.DataFrame(dict_stats_patient).T], axis=1)
    return stats_per_patient

def get_nb_msrt(msr_per_patient):
    return msr_per_patient.size()


def extract_stats_missing(parquet_file):
    df = pd.read_parquet(parquet_file)
    stats_per_patient = pd.DataFrame()
    for var in BASELINE_INTERVAL.keys():
        msr_per_patient = df[['patientid', var, 'datetime']].dropna().groupby('patientid')
        stats_per_patient[f'nb_msrt_{var}'] = get_nb_msrt(msr_per_patient)
        stats_per_patient = get_stats_interval_per_patient(msr_per_patient, stats_per_patient, var)
    return stats_per_patient


def main():
    stats_patients = pd.DataFrame()
    for filename in os.listdir(MERGED_PATH):
        if os.path.splitext(filename)[1] == ".parquet":
            stats_patients = pd.concat(
                [stats_patients, extract_stats_missing(os.path.join(MERGED_PATH, filename))]
            )

    stats_patients.to_csv(os.path.join(OUTPUT_PATH, "msrt_stats_patients.csv"))