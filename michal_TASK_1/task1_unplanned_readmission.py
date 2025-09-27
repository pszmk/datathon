from __future__ import annotations
import numpy as np
import pandas as pd

# ICD-10 groupings used for the “complication” rule: T80–T88, Y83–Y84
COMPLICATION_PREFIXES = tuple([f"T8{i}" for i in range(0, 9)] + ["Y83", "Y84"])

# COVID-19 exclusions (dot removed)
COVID_CODES = {"U071", "U072"}  # U07.1, U07.2

# Cancer treatment primary diagnosis codes (as per brief: Z51.0, Z51.11, Z51.12)
# Implemented as exact Z510 or Z511* prefixes after dot removal.
CANCER_TREATMENT_EQ = {"Z510"}
CANCER_TREATMENT_PREFIX = ("Z511",)

# --- Admission keywords (ENGLISH-ONLY, per the PDF) ---
# PDF lists "emergency admission" explicitly; include common English synonyms used in data.
EMERGENCY_KEYWORDS = (
    "emergency admission", "emergency", "urgent", "unscheduled",
    "accident & emergency", "a&e", "er admission",
)

# Things that clearly indicate NON-emergency / planned context
PLANNED_KEYWORDS = (
    "planned admission based on a referral",
    "planned", "elective", "by referral", "based on a referral",
)

# Explicit transfer at admission — do not classify as emergency for our logic
TRANSFER_IN_ADM_KEYWORDS = ("transfer from another hospital",)

# --- Discharge keywords (ENGLISH-ONLY, exactly as in PDF) ---
DISCHARGE_STANDARD_KEYWORDS = (
    "completion of the therapeutic or diagnostic process",
    "referral for further treatment in an outpatient setting",
)

DISCHARGE_TRANSFER_KEYWORDS = ("referral for further treatment in another hospital",)

DISCHARGE_DEATH_KEYWORDS = ("patient's death", "patient died", "death")

DISCHARGE_OWN_REQUEST_KEYWORDS = ("discharge on own request",)

DATE_COLS_ADM = ["stay_start_date", "stay_end_date"]


def _norm_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().lower().split())


def _norm_code(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper().replace(".", "")


def is_emergency_admission(proc) -> bool:
    """Detect emergency/unscheduled admission (English-only, per PDF)."""
    s = _norm_text(proc)
    # If it looks planned or a transfer-in, treat as non-emergency
    if any(k in s for k in PLANNED_KEYWORDS):
        return False
    if any(k in s for k in TRANSFER_IN_ADM_KEYWORDS):
        return False
    return any(k in s for k in EMERGENCY_KEYWORDS)


def discharge_category(discharge_proc) -> str:
    """Derive discharge category (English-only keywords from the PDF)."""
    s = _norm_text(discharge_proc)
    if any(k in s for k in DISCHARGE_DEATH_KEYWORDS):
        return "death"
    if any(k in s for k in DISCHARGE_TRANSFER_KEYWORDS):
        return "transfer"
    if any(k in s for k in DISCHARGE_OWN_REQUEST_KEYWORDS):
        return "own_request"
    if any(k in s for k in DISCHARGE_STANDARD_KEYWORDS):
        return "standard"
    return "other"


def build_hospital_episodes(adm: pd.DataFrame) -> pd.DataFrame:
    """Collapse ward/clinic stays to hospitalization episodes and add temporal links."""
    df = adm.copy()
    for c in DATE_COLS_ADM:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    first = (
        df.sort_values(["patient_id", "hospitalization_id", "stay_start_date"])
        .groupby(["patient_id", "hospitalization_id"], as_index=False)
        .first()
    )
    last = (
        df.sort_values(["patient_id", "hospitalization_id", "stay_end_date"])
        .groupby(["patient_id", "hospitalization_id"], as_index=False)
        .last()
    )

    episodes = last[
        [
            "patient_id",
            "hospitalization_id",
            "stay_id",
            "stay_end_date",
            "hospital_discharge_procedure",
            "patient_age",
            "patient_sex",
        ]
        + (["did_patient_die"] if "did_patient_die" in last.columns else [])
    ].rename(columns={"stay_id": "last_stay_id", "stay_end_date": "episode_end"})

    episodes = episodes.merge(
        first[
            [
                "patient_id",
                "hospitalization_id",
                "stay_start_date",
                "hospital_admission_procedure",
            ]
        ].rename(columns={"stay_start_date": "episode_start"}),
        on=["patient_id", "hospitalization_id"],
        how="left",
    )

    episodes["discharge_category"] = episodes["hospital_discharge_procedure"].apply(
        discharge_category
    )
    episodes["is_emergency_admission"] = episodes[
        "hospital_admission_procedure"
    ].apply(is_emergency_admission)

    # Ensure did_patient_die consistent with discharge
    if "did_patient_die" not in episodes.columns:
        episodes["did_patient_die"] = episodes["discharge_category"].eq("death").astype(int)
    else:
        episodes["did_patient_die"] = (
            episodes["did_patient_die"].fillna(0).astype(int).clip(0, 1)
        )
        episodes.loc[episodes["discharge_category"].eq("death"), "did_patient_die"] = 1

    # Temporal linking across episodes for each patient
    episodes = episodes.sort_values(["patient_id", "episode_start", "episode_end"])
    episodes["prev_episode_end"] = episodes.groupby("patient_id")["episode_end"].shift(1)
    episodes["next_episode_start"] = episodes.groupby("patient_id")["episode_start"].shift(-1)
    episodes["next_hospitalization_id"] = episodes.groupby("patient_id")[
        "hospitalization_id"
    ].shift(-1)
    episodes["prev_discharge_category"] = episodes.groupby("patient_id")[
        "discharge_category"
    ].shift(1)

    # Intervals
    episodes["days_from_prev"] = (
        episodes["episode_start"] - episodes["prev_episode_end"]
    ).dt.days
    episodes["days_to_next"] = (episodes["next_episode_start"] - episodes["episode_end"]).dt.days

    # Mark if current episode is a readmission (<=30d) only when PREVIOUS discharge was standard
    episodes["prev_discharge_standard"] = episodes["prev_discharge_category"].eq("standard")
    episodes["is_current_readmission_any"] = (
        episodes["days_from_prev"].notna()
        & (episodes["days_from_prev"] >= 0)
        & (episodes["days_from_prev"] <= 30)
        & episodes["prev_discharge_standard"].fillna(False)
    )

    return episodes.reset_index(drop=True)


def aggregate_diagnoses_to_episode(diag: pd.DataFrame, adm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate diagnosis info to hospitalization level."""
    if diag.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "hospitalization_id",
                "count_of_comorbidities",
                "has_complication_dx",
                "has_covid_dx",
                "primary_is_psychiatry",
                "primary_is_cancer_treatment",
                "primary_dx_code",
            ]
        )

    d = diag.copy()
    d["code_norm"] = d["diagnosis_code"].map(_norm_code)

    for col in ("kind_of_diagnosis", "type_of_diagnosis"):
        if col in d.columns:
            d[col] = d[col].fillna("").astype(str)

    # English-only normalization, per PDF: "Main / Primary", "Comorbid", "Secondary"
    d["kind_norm"] = (
        d["kind_of_diagnosis"]
        .str.lower()
        .str.replace(r"[^a-z/ ]", "", regex=True)
        .str.strip()
    )
    d["type_norm"] = d["type_of_diagnosis"].str.lower().str.strip()

    d["is_primary"] = d["kind_norm"].str.contains("primary") | d["kind_norm"].str.contains("main")
    d["is_comorbid"] = d["kind_norm"].str.contains("comorbid")

    # Flags
    d["flag_complication"] = d["code_norm"].map(lambda x: any(x.startswith(p) for p in COMPLICATION_PREFIXES))
    d["flag_covid"] = d["code_norm"].isin(COVID_CODES)
    d["flag_primary_psychiatry"] = d["is_primary"] & d["code_norm"].str.startswith("F")

    # Cancer treatment: primary AND (Z510 exact OR Z511x prefix)
    d["flag_primary_cancer_treatment"] = d["is_primary"] & (
        d["code_norm"].isin(CANCER_TREATMENT_EQ)
        | d["code_norm"].str.startswith(CANCER_TREATMENT_PREFIX)
    )

    d["primary_dx_code_candidate"] = np.where(d["is_primary"], d["code_norm"], np.nan)

    # Map diagnoses to hospitalization
    d = d.merge(adm[["stay_id", "patient_id", "hospitalization_id"]], on="stay_id", how="left")

    agg = d.groupby(["patient_id", "hospitalization_id"]).agg(
        count_of_comorbidities=("is_comorbid", "sum"),
        has_complication_dx=("flag_complication", "any"),
        has_covid_dx=("flag_covid", "any"),
        primary_is_psychiatry=("flag_primary_psychiatry", "any"),
        primary_is_cancer_treatment=("flag_primary_cancer_treatment", "any"),
        primary_dx_code=("primary_dx_code_candidate", "first"),
    )
    # Fallback primary code: take any code if none marked as primary
    fallback = d.groupby(["patient_id", "hospitalization_id"]).agg(_any_dx=("code_norm", "first"))
    agg["primary_dx_code"] = agg["primary_dx_code"].fillna(fallback["_any_dx"])

    out = agg.reset_index()
    # Ensure integer type for comorbidity count
    out["count_of_comorbidities"] = out["count_of_comorbidities"].astype(int)
    return out


def build_index_cohort_with_labels(admissions_stays: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Return index hospitalizations with the is_unplanned_readmission label (English-only rules)."""
    episodes = build_hospital_episodes(admissions_stays)
    diag_epi = aggregate_diagnoses_to_episode(diagnoses, admissions_stays)

    epi = episodes.merge(diag_epi, on=["patient_id", "hospitalization_id"], how="left")

    # Fill diagnosis-derived booleans
    for col, default in [
        ("count_of_comorbidities", 0),
        ("has_complication_dx", False),
        ("has_covid_dx", False),
        ("primary_is_psychiatry", False),
        ("primary_is_cancer_treatment", False),
    ]:
        epi[col] = epi[col].fillna(default)

    # Cohort (denominator) per PDF:
    # - age >= 18
    # - standard discharge
    # - not died
    # - exclude psychiatry, COVID-19, cancer treatment primaries
    # - exclude any admission that is already a readmission (<=30d after a previous STANDARD discharge)
    mask_age = epi["patient_age"].fillna(0) >= 18
    mask_standard_discharge = epi["discharge_category"].eq("standard")
    mask_not_dead = epi["did_patient_die"].fillna(0).astype(int).eq(0)
    mask_exclude_psych = ~epi["primary_is_psychiatry"]
    mask_exclude_covid = ~epi["has_covid_dx"]
    mask_exclude_cancer_treat = ~epi["primary_is_cancer_treatment"]
    mask_not_current_readm = ~epi["is_current_readmission_any"].fillna(False)

    cohort = epi[
        mask_age
        & mask_standard_discharge
        & mask_not_dead
        & mask_exclude_psych
        & mask_exclude_covid
        & mask_exclude_cancer_treat
        & mask_not_current_readm
    ].copy()

    # Features from the NEXT episode for unplanned logic
    next_feats = epi[
        ["patient_id", "hospitalization_id", "is_emergency_admission", "has_complication_dx"]
    ].rename(
        columns={
            "is_emergency_admission": "next_is_emergency_admission",
            "has_complication_dx": "next_has_complication_dx",
        }
    )

    cohort = cohort.merge(
        next_feats,
        left_on=["patient_id", "next_hospitalization_id"],
        right_on=["patient_id", "hospitalization_id"],
        how="left",
        suffixes=("", "_nextrow"),
    )
    if "hospitalization_id_nextrow" in cohort.columns:
        cohort = cohort.drop(columns=["hospitalization_id_nextrow"])

    # Readmission within 30 days (non-negative guard)
    cohort["is_readmission_30d"] = (
        cohort["days_to_next"].notna()
        & (cohort["days_to_next"] >= 0)
        & (cohort["days_to_next"] <= 30)
    )

    # Unplanned readmission per PDF:
    # next admission is emergency OR has a complication code
    cohort["is_unplanned_readmission"] = (
    cohort["is_readmission_30d"].astype("boolean").fillna(False)
    & (
        cohort["next_is_emergency_admission"].astype("boolean").fillna(False)
        | cohort["next_has_complication_dx"].astype("boolean").fillna(False)
    )
).astype("int8")


    out_cols = [
        "patient_id",
        "hospitalization_id",
        "episode_start",
        "episode_end",
        "patient_age",
        "patient_sex",
        "hospital_admission_procedure",
        "hospital_discharge_procedure",
        "discharge_category",
        "count_of_comorbidities",
        "primary_dx_code",
        "primary_is_psychiatry",
        "primary_is_cancer_treatment",
        "has_covid_dx",
        "next_hospitalization_id",
        "days_to_next",
        "is_readmission_30d",
        "next_is_emergency_admission",
        "next_has_complication_dx",
        "is_unplanned_readmission",
    ]
    out_cols = [c for c in out_cols if c in cohort.columns]
    return (
        cohort[out_cols]
        .sort_values(["patient_id", "episode_start"])
        .reset_index(drop=True)
    )
