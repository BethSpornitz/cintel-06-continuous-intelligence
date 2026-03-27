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

    dates = dashboard_df["visit_datetime"].to_list()
    rolling_wait = dashboard_df["rolling_avg_wait_time"].to_list()
    rolling_satisfaction = dashboard_df["rolling_avg_satisfaction"].to_list()
    anomaly_count = dashboard_df["anomaly_count"].to_list()

    MIN_AVG_SATISFACTION = 6.5

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # ----------------------------------------------------
    # VISUAL 1A: Rolling Average Wait Time with Alert Points
    # ----------------------------------------------------
    axes[0].plot(
        dates,
        rolling_wait,
        color="green",
        linewidth=2,
        label="Rolling Avg Wait Time",
    )

    axes[0].axhline(
        y=MAX_AVG_WAIT_TIME,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Wait Time Threshold",
    )

    wait_yellow = []
    wait_orange = []
    wait_red = []

    for d, v in zip(dates, rolling_wait, strict=False):
        if v > MAX_AVG_WAIT_TIME * 1.20:
            wait_red.append((d, v))
        elif v > MAX_AVG_WAIT_TIME * 1.10:
            wait_orange.append((d, v))
        elif v > MAX_AVG_WAIT_TIME:
            wait_yellow.append((d, v))

    if wait_yellow:
        wx, wy = zip(*wait_yellow, strict=False)
        axes[0].scatter(wx, wy, color="gold", s=25, zorder=3, label="Warning")
    if wait_orange:
        ox, oy = zip(*wait_orange, strict=False)
        axes[0].scatter(ox, oy, color="orange", s=30, zorder=3, label="Elevated")
    if wait_red:
        rx, ry = zip(*wait_red, strict=False)
        axes[0].scatter(rx, ry, color="red", s=35, zorder=3, label="Critical")

    axes[0].set_title("Rolling Average Wait Time")
    axes[0].set_ylabel("Minutes")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # ----------------------------------------------------
    # VISUAL 1B: Rolling Average Satisfaction with Alert Points
    # ----------------------------------------------------
    axes[1].plot(
        dates,
        rolling_satisfaction,
        color="green",
        linewidth=2,
        label="Rolling Avg Satisfaction",
    )

    axes[1].axhline(
        y=MIN_AVG_SATISFACTION,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Satisfaction Threshold",
    )

    sat_yellow = []
    sat_orange = []
    sat_red = []

    for d, v in zip(dates, rolling_satisfaction, strict=False):
        if v < MIN_AVG_SATISFACTION * 0.80:
            sat_red.append((d, v))
        elif v < MIN_AVG_SATISFACTION * 0.90:
            sat_orange.append((d, v))
        elif v < MIN_AVG_SATISFACTION:
            sat_yellow.append((d, v))

    if sat_yellow:
        sx, sy = zip(*sat_yellow, strict=False)
        axes[1].scatter(sx, sy, color="gold", s=25, zorder=3, label="Warning")
    if sat_orange:
        sox, soy = zip(*sat_orange, strict=False)
        axes[1].scatter(sox, soy, color="orange", s=30, zorder=3, label="Elevated")
    if sat_red:
        srx, sry = zip(*sat_red, strict=False)
        axes[1].scatter(srx, sry, color="red", s=35, zorder=3, label="Critical")

    axes[1].set_title("Rolling Average Patient Satisfaction")
    axes[1].set_ylabel("Satisfaction")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # ----------------------------------------------------
    # VISUAL 1C: System Anomalies Over Time with Alert Points
    # ----------------------------------------------------
    axes[2].plot(
        dates,
        anomaly_count,
        color="green",
        linewidth=2,
        label="Anomaly Count",
    )

    anom_yellow = []
    anom_orange = []
    anom_red = []

    for d, v in zip(dates, anomaly_count, strict=False):
        if v >= 3:
            anom_red.append((d, v))
        elif v == 2:
            anom_orange.append((d, v))
        elif v == 1:
            anom_yellow.append((d, v))

    if anom_yellow:
        ayx, ayy = zip(*anom_yellow, strict=False)
        axes[2].scatter(ayx, ayy, color="gold", s=25, zorder=3, label="1 Anomaly")
    if anom_orange:
        aox, aoy = zip(*anom_orange, strict=False)
        axes[2].scatter(aox, aoy, color="orange", s=30, zorder=3, label="2 Anomalies")
    if anom_red:
        arx, ary = zip(*anom_red, strict=False)
        axes[2].scatter(arx, ary, color="red", s=35, zorder=3, label="3+ Anomalies")

    axes[2].set_title("System Anomalies Over Time")
    axes[2].set_ylabel("Anomaly Count")
    axes[2].set_xlabel("Visit Date")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

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

    wait_bins = satisfaction_bin_df["wait_time_bin"].to_list()
    avg_satisfaction = satisfaction_bin_df["avg_patient_satisfaction"].to_list()

    plt.figure(figsize=(10, 6))
    plt.plot(
        wait_bins,
        avg_satisfaction,
        color="green",
        linewidth=2,
        marker="o",
        label="Avg Satisfaction",
    )

    bin_yellow = []
    bin_orange = []
    bin_red = []

    for x, y in zip(wait_bins, avg_satisfaction, strict=False):
        if y < MIN_AVG_SATISFACTION * 0.80:
            bin_red.append((x, y))
        elif y < MIN_AVG_SATISFACTION * 0.90:
            bin_orange.append((x, y))
        elif y < MIN_AVG_SATISFACTION:
            bin_yellow.append((x, y))

    if bin_yellow:
        byx, byy = zip(*bin_yellow, strict=False)
        plt.scatter(byx, byy, color="gold", s=25, zorder=3, label="Warning")
    if bin_orange:
        box, boy = zip(*bin_orange, strict=False)
        plt.scatter(box, boy, color="orange", s=30, zorder=3, label="Elevated")
    if bin_red:
        brx, bry = zip(*bin_red, strict=False)
        plt.scatter(brx, bry, color="red", s=35, zorder=3, label="Critical")

    plt.axhline(
        y=MIN_AVG_SATISFACTION,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Satisfaction Threshold",
    )

    plt.title("Average Patient Satisfaction by Wait Time")
    plt.xlabel("Wait Time Bin (minutes)")
    plt.ylabel("Average Satisfaction")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        ARTIFACTS_DIR / "wait_time_vs_satisfaction_binned_bethspornitz.png",
        dpi=300,
    )
    plt.close()

    LOG.info("STEP 6 complete: visualization artifacts created")

    LOG.info("========================")
    LOG.info("Pipeline executed successfully!")
    LOG.info("========================")
    LOG.info("END main()")


if __name__ == "__main__":
    main()
