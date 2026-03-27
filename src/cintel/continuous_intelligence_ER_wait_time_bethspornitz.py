"""
continuous_intelligence_ER_wait_time_bethspornitz.py - Custom Continuous Intelligence pipeline.

Author: Beth Spornitz
Date: 2026-03

Emergency Department Wait Time Data

Purpose
- Read ED visit data from a CSV file
- Design monitoring signals from wait-time and care variables
- Detect operational anomalies
- Summarize current system state
- Save summary and visualization artifacts

Terminal command to run from repo root:

    uv run python -m cintel.continuous_intelligence_ER_wait_time_bethspornitz
"""

# === DECLARE IMPORTS ===

import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import polars as pl
from datafun_toolkit.logger import get_logger, log_header, log_path

# === CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("P6", level="DEBUG")

# === DEFINE GLOBAL PATHS ===

ROOT_DIR: Final[Path] = Path.cwd()
DATA_DIR: Final[Path] = ROOT_DIR / "data"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"

DATA_FILE: Final[Path] = DATA_DIR / "er_wait_time_bethspornitz.csv"
OUTPUT_FILE: Final[Path] = ARTIFACTS_DIR / "ed_system_assessment_bethspornitz.csv"
PLOT_FILE: Final[Path] = ARTIFACTS_DIR / "ed_wait_time_trend_bethspornitz.png"

# === DEFINE THRESHOLDS ===

MAX_AVG_WAIT_TIME: Final[float] = 120.0
MIN_SATISFACTION: Final[float] = 2.5
MAX_LEFT_WITHOUT_BEING_SEEN_RATE: Final[float] = 0.08
MIN_NURSE_PATIENT_RATIO: Final[float] = 3.0


def main() -> None:
    """Run the custom ED monitoring pipeline."""
    log_header(LOG, "CINTEL")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    log_path(LOG, "ROOT_DIR", ROOT_DIR)
    log_path(LOG, "DATA_FILE", DATA_FILE)
    log_path(LOG, "OUTPUT_FILE", OUTPUT_FILE)
    log_path(LOG, "PLOT_FILE", PLOT_FILE)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path(LOG, "ARTIFACTS_DIR", ARTIFACTS_DIR)

    # ----------------------------------------------------
    # STEP 1: READ ED DATA
    # ----------------------------------------------------
    df = pl.read_csv(DATA_FILE)

    LOG.info(f"STEP 1. Loaded {df.height} ED visit records")

    # Convert visit date to datetime and sort
    df = df.with_columns(
        pl.col("Visit Date")
        .str.strptime(pl.Datetime, strict=False)
        .alias("visit_datetime")
    ).sort("visit_datetime")

    # ----------------------------------------------------
    # STEP 2: DESIGN SIGNALS
    # ----------------------------------------------------
    LOG.info("STEP 2. Designing monitoring signals...")

    df = df.with_columns(
        [
            (pl.col("Patient Outcome") == "Left Without Being Seen")
            .cast(pl.Int64)
            .alias("left_without_being_seen_flag"),
            (pl.col("Total Wait Time (min)") > 180)
            .cast(pl.Int64)
            .alias("high_wait_flag"),
        ]
    )

    # rolling signals
    df = df.with_columns(
        [
            pl.col("Total Wait Time (min)")
            .rolling_mean(window_size=50, min_samples=10)
            .alias("rolling_avg_wait_time"),
            pl.col("Total Wait Time (min)")
            .rolling_std(window_size=50, min_samples=10)
            .alias("rolling_std_wait_time"),
            pl.col("Patient Satisfaction")
            .rolling_mean(window_size=50, min_samples=10)
            .alias("rolling_avg_satisfaction"),
            pl.col("left_without_being_seen_flag")
            .rolling_mean(window_size=50, min_samples=10)
            .alias("rolling_left_without_being_seen_rate"),
            pl.col("Nurse-to-Patient Ratio")
            .rolling_mean(window_size=50, min_samples=10)
            .alias("rolling_avg_nurse_patient_ratio"),
        ]
    )

    # ----------------------------------------------------
    # STEP 3: DETECT ANOMALIES
    # ----------------------------------------------------
    LOG.info("STEP 3. Checking for anomalies...")

    df = df.with_columns(
        [
            (pl.col("rolling_avg_wait_time") > MAX_AVG_WAIT_TIME).alias(
                "wait_time_anomaly"
            ),
            (pl.col("rolling_avg_satisfaction") < MIN_SATISFACTION).alias(
                "satisfaction_anomaly"
            ),
            (
                pl.col("rolling_left_without_being_seen_rate")
                > MAX_LEFT_WITHOUT_BEING_SEEN_RATE
            ).alias("lwbs_anomaly"),
            (pl.col("rolling_avg_nurse_patient_ratio") < MIN_NURSE_PATIENT_RATIO).alias(
                "staffing_anomaly"
            ),
        ]
    )

    df = df.with_columns(
        (
            pl.col("wait_time_anomaly").cast(pl.Int64)
            + pl.col("satisfaction_anomaly").cast(pl.Int64)
            + pl.col("lwbs_anomaly").cast(pl.Int64)
            + pl.col("staffing_anomaly").cast(pl.Int64)
        ).alias("anomaly_count")
    )

    anomalies_df = df.filter(pl.col("anomaly_count") > 0)
    LOG.info(f"STEP 3. Anomalous rows detected: {anomalies_df.height}")

    # ----------------------------------------------------
    # STEP 4: DETERMINE SYSTEM STATE
    # ----------------------------------------------------
    LOG.info("STEP 4. Determining overall system state...")

    summary_df = df.select(
        [
            pl.col("Total Wait Time (min)").mean().alias("avg_total_wait_time"),
            pl.col("Patient Satisfaction").mean().alias("avg_patient_satisfaction"),
            pl.col("Nurse-to-Patient Ratio").mean().alias("avg_nurse_patient_ratio"),
            pl.col("left_without_being_seen_flag")
            .mean()
            .alias("avg_left_without_being_seen_rate"),
            pl.col("high_wait_flag").mean().alias("high_wait_rate"),
            pl.col("rolling_avg_wait_time").mean().alias("avg_rolling_wait_time"),
            pl.col("rolling_avg_satisfaction").mean().alias("avg_rolling_satisfaction"),
            pl.col("rolling_left_without_being_seen_rate")
            .mean()
            .alias("avg_rolling_lwbs_rate"),
            pl.col("rolling_avg_nurse_patient_ratio")
            .mean()
            .alias("avg_rolling_nurse_patient_ratio"),
            pl.col("anomaly_count").max().alias("max_anomaly_count"),
            pl.col("anomaly_count").mean().alias("avg_anomaly_count"),
        ]
    )

    summary_df = summary_df.with_columns(
        pl.when(
            (pl.col("max_anomaly_count") >= 2)
            | (
                (pl.col("avg_rolling_wait_time") > MAX_AVG_WAIT_TIME)
                & (pl.col("avg_rolling_nurse_patient_ratio") < MIN_NURSE_PATIENT_RATIO)
            )
        )
        .then(pl.lit("CRITICAL"))
        .when(pl.col("avg_anomaly_count") >= 1)
        .then(pl.lit("WARNING"))
        .otherwise(pl.lit("STABLE"))
        .alias("system_state")
    )

    LOG.info("STEP 4. System state assessment completed")

    # ----------------------------------------------------
    # STEP 5: SAVE SUMMARY ARTIFACT
    # ----------------------------------------------------
    summary_df.write_csv(OUTPUT_FILE)
    LOG.info(f"STEP 5. Wrote summary artifact: {OUTPUT_FILE}")

    # ----------------------------------------------------
    # STEP 6: SAVE VISUALIZATION ARTIFACTS
    # ----------------------------------------------------
    LOG.info("STEP 6. Creating visualization artifacts...")

    # ----------------------------------------------------
    # VISUAL 1: ED System Monitoring Dashboard
    # ----------------------------------------------------
    dashboard_df = df.select(
        [
            "visit_datetime",
            "rolling_avg_wait_time",
            "rolling_avg_satisfaction",
            "anomaly_count",
        ]
    ).drop_nulls()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Top panel: Rolling Average Wait Time
    axes[0].plot(
        dashboard_df["visit_datetime"].to_list(),
        dashboard_df["rolling_avg_wait_time"].to_list(),
    )
    axes[0].axhline(y=MAX_AVG_WAIT_TIME, linestyle="--")
    axes[0].set_title("Rolling Average Wait Time")
    axes[0].set_ylabel("Minutes")

    # Middle panel: Rolling Average Patient Satisfaction
    axes[1].plot(
        dashboard_df["visit_datetime"].to_list(),
        dashboard_df["rolling_avg_satisfaction"].to_list(),
    )
    axes[1].set_title("Rolling Average Patient Satisfaction")
    axes[1].set_ylabel("Satisfaction")

    # Bottom panel: System Anomalies Over Time
    axes[2].plot(
        dashboard_df["visit_datetime"].to_list(),
        dashboard_df["anomaly_count"].to_list(),
    )
    axes[2].set_title("System Anomalies Over Time")
    axes[2].set_ylabel("Anomaly Count")
    axes[2].set_xlabel("Visit Date")

    plt.suptitle("ED System Monitoring Dashboard")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "ed_dashboard_bethspornitz.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # VISUAL 2: Average Patient Satisfaction by Wait Time
    # ----------------------------------------------------
    satisfaction_bin_df = (
        df.with_columns(
            (pl.col("Total Wait Time (min)") // 10 * 10).alias("wait_time_bin")
        )
        .group_by("wait_time_bin")
        .agg(pl.col("Patient Satisfaction").mean().alias("avg_patient_satisfaction"))
        .sort("wait_time_bin")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        satisfaction_bin_df["wait_time_bin"].to_list(),
        satisfaction_bin_df["avg_patient_satisfaction"].to_list(),
        marker="o",
    )
    plt.title("Average Patient Satisfaction by Wait Time")
    plt.xlabel("Wait Time Bin (minutes)")
    plt.ylabel("Average Satisfaction")
    plt.tight_layout()
    plt.savefig(
        ARTIFACTS_DIR / "wait_time_vs_satisfaction_binned_bethspornitz.png",
        dpi=300,
    )
    plt.close()

    LOG.info("STEP 6. Wrote visualization artifacts to the artifacts folder")

    LOG.info("========================")
    LOG.info("Pipeline executed successfully!")
    LOG.info("========================")
    LOG.info("END main()")


if __name__ == "__main__":
    main()
