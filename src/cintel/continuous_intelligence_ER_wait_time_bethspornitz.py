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

    dashboard_df = df.select(
        [
            "visit_datetime",
            "rolling_avg_wait_time",
            "rolling_avg_satisfaction",
            "rolling_left_without_being_seen_rate",
            "rolling_avg_nurse_patient_ratio",
            "anomaly_count",
        ]
    ).drop_nulls()

    dates = dashboard_df["visit_datetime"].to_list()
    rolling_wait = dashboard_df["rolling_avg_wait_time"].to_list()
    rolling_satisfaction = dashboard_df["rolling_avg_satisfaction"].to_list()
    anomaly_count = dashboard_df["anomaly_count"].to_list()

    MIN_AVG_SATISFACTION = 6.5

    latest_wait = rolling_wait[-1]
    latest_satisfaction = rolling_satisfaction[-1]
    latest_anomalies = anomaly_count[-1]

    if latest_anomalies >= 3:
        current_status = "CRITICAL"
    elif latest_anomalies >= 1:
        current_status = "WARNING"
    else:
        current_status = "STABLE"

    def get_status_color(
        value: float, threshold: float, higher_is_bad: bool = True
    ) -> str:
        """Return executive dashboard color by severity."""
        if higher_is_bad:
            if value >= threshold * 1.15:
                return "red"
            if value >= threshold:
                return "orange"
            return "green"
        else:
            if value <= threshold * 0.85:
                return "red"
            if value <= threshold:
                return "orange"
            return "green"

    wait_color = get_status_color(latest_wait, MAX_AVG_WAIT_TIME, higher_is_bad=True)
    satisfaction_color = get_status_color(
        latest_satisfaction, MIN_AVG_SATISFACTION, higher_is_bad=False
    )

    if latest_anomalies >= 3:
        anomaly_color = "red"
    elif latest_anomalies == 2:
        anomaly_color = "orange"
    elif latest_anomalies == 1:
        anomaly_color = "gold"
    else:
        anomaly_color = "green"

    if current_status == "CRITICAL":
        state_color = "red"
    elif current_status == "WARNING":
        state_color = "orange"
    else:
        state_color = "green"

    # ----------------------------------------------------
    # VISUAL 1: Executive Monitoring Dashboard
    # ----------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        height_ratios=[1.2, 2.2, 2.2, 0.8],
        hspace=0.5,
        wspace=0.35,
    )

    # KPI cards
    ax_kpi1 = fig.add_subplot(gs[0, 0])
    ax_kpi2 = fig.add_subplot(gs[0, 1])
    ax_kpi3 = fig.add_subplot(gs[0, 2])
    ax_kpi4 = fig.add_subplot(gs[0, 3])

    # Trend charts
    ax_wait = fig.add_subplot(gs[1, :])
    ax_sat = fig.add_subplot(gs[2, :])

    # Alert strip
    ax_alert = fig.add_subplot(gs[3, :])

    for ax in [ax_kpi1, ax_kpi2, ax_kpi3, ax_kpi4]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def draw_kpi_card(
        ax, title: str, value_text: str, subtitle: str, color: str
    ) -> None:
        """Draw a simple executive KPI card."""
        ax.set_facecolor("#f5f5f5")
        ax.text(
            0.5, 0.78, title, ha="center", va="center", fontsize=11, fontweight="bold"
        )
        ax.text(
            0.5,
            0.48,
            value_text,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=color,
        )
        ax.text(0.5, 0.18, subtitle, ha="center", va="center", fontsize=10)

    draw_kpi_card(
        ax_kpi1,
        "Rolling Wait Time",
        f"{latest_wait:.1f} min",
        f"Threshold: {MAX_AVG_WAIT_TIME:.0f}",
        wait_color,
    )
    draw_kpi_card(
        ax_kpi2,
        "Rolling Satisfaction",
        f"{latest_satisfaction:.2f}",
        f"Threshold: {MIN_AVG_SATISFACTION:.1f}",
        satisfaction_color,
    )
    draw_kpi_card(
        ax_kpi3,
        "Current Anomaly Level",
        f"{int(latest_anomalies)}",
        "Combined active signals",
        anomaly_color,
    )
    draw_kpi_card(
        ax_kpi4,
        "System State",
        current_status,
        "Current overall status",
        state_color,
    )

    # ----------------------------------------------------
    # WAIT TIME CHART WITH SHADED ALERT PERIODS
    # ----------------------------------------------------
    ax_wait.plot(
        dates,
        rolling_wait,
        color="green",
        linewidth=2,
        label="Rolling Avg Wait Time",
    )
    ax_wait.axhline(
        y=MAX_AVG_WAIT_TIME,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Wait Threshold",
    )

    above_wait = [v > MAX_AVG_WAIT_TIME for v in rolling_wait]
    for i in range(len(dates) - 1):
        if above_wait[i]:
            ax_wait.axvspan(dates[i], dates[i + 1], color="red", alpha=0.18)

    ax_wait.set_title("Rolling Average Wait Time with Alert Periods")
    ax_wait.set_ylabel("Minutes")
    ax_wait.grid(alpha=0.3)
    ax_wait.legend(loc="upper left")

    # ----------------------------------------------------
    # SATISFACTION CHART WITH SHADED ALERT PERIODS
    # ----------------------------------------------------
    ax_sat.plot(
        dates,
        rolling_satisfaction,
        color="green",
        linewidth=2,
        label="Rolling Avg Satisfaction",
    )
    ax_sat.axhline(
        y=MIN_AVG_SATISFACTION,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Satisfaction Threshold",
    )

    below_sat = [v < MIN_AVG_SATISFACTION for v in rolling_satisfaction]
    for i in range(len(dates) - 1):
        if below_sat[i]:
            ax_sat.axvspan(dates[i], dates[i + 1], color="red", alpha=0.18)

    ax_sat.set_title("Rolling Average Patient Satisfaction with Alert Periods")
    ax_sat.set_ylabel("Satisfaction")
    ax_sat.grid(alpha=0.3)
    ax_sat.legend(loc="upper left")

    # ----------------------------------------------------
    # ALERT TIMELINE STRIP
    # ----------------------------------------------------
    status_levels = []
    for count in anomaly_count:
        if count >= 3:
            status_levels.append(3)
        elif count == 2:
            status_levels.append(2)
        elif count == 1:
            status_levels.append(1)
        else:
            status_levels.append(0)

    ax_alert.imshow(
        [status_levels],
        aspect="auto",
        interpolation="nearest",
        extent=[0, len(status_levels), 0, 1],
        cmap="RdYlGn_r",
    )

    ax_alert.set_yticks([])
    tick_positions = list(range(0, len(dates), max(1, len(dates) // 8)))
    tick_labels = [dates[i].strftime("%Y-%m-%d") for i in tick_positions]
    ax_alert.set_xticks(tick_positions)
    ax_alert.set_xticklabels(tick_labels, rotation=0)
    ax_alert.set_title("Alert Timeline: Green=Normal, Yellow=1, Orange=2, Red=3+")
    ax_alert.set_xlabel("Visit Date")

    plt.suptitle("ED Executive Monitoring Dashboard", fontsize=16, fontweight="bold")
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
        marker="o",
        linewidth=2,
        color="green",
        label="Avg Satisfaction",
    )
    plt.axhline(
        y=MIN_AVG_SATISFACTION,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Satisfaction Threshold",
    )
    plt.title("Average Patient Satisfaction by Wait Time Bin")
    plt.xlabel("Wait Time Bin (minutes)")
    plt.ylabel("Average Satisfaction")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        ARTIFACTS_DIR / "wait_time_vs_satisfaction_binned_bethspornitz.png",
        dpi=300,
    )
    plt.close()

    LOG.info("STEP 6 complete: visualization artifacts created")
    LOG.info("STEP 6 complete: visualization artifacts created")

    LOG.info("========================")
    LOG.info("Pipeline executed successfully!")
    LOG.info("========================")
    LOG.info("END main()")


if __name__ == "__main__":
    main()
