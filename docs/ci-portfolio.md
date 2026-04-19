# Continuous Intelligence Portfolio

Beth Spornitz

2026-04

This page summarizes my work across multiple modules in a Continuous Intelligence course. Each module focused on a different technique used to monitor, analyze, and interpret system behavior.

---

## 1. Professional Project

### Repository Link
https://github.com/BethSpornitz/cintel-06-continuous-intelligence

### Brief Overview of Project Tools and Choices
Across these modules, I used Python with Polars to implement continuous intelligence techniques including anomaly detection, signal design, rolling monitoring, drift detection, and system state assessment. The final project applied these techniques to a healthcare Emergency Department (ED) dataset to evaluate system performance and patient flow.

---

## 2. Anomaly Detection

### Repository Link
https://github.com/BethSpornitz/cintel-02-static-anomalies

### Techniques
Anomalies were detected using threshold-based rules applied to variables such as age and height. Values outside of expected ranges were flagged and categorized with specific anomaly reasons.

### Artifacts
https://github.com/BethSpornitz/cintel-02-static-anomalies/tree/main/artifacts

The output dataset includes records identified as anomalies along with explanations for why they were flagged.

### Insights
This module demonstrated how simple rule-based techniques can quickly identify data quality issues, ensuring that unreliable data does not impact downstream analysis.

---

## 3. Signal Design

### Repository Link
https://github.com/BethSpornitz/cintel-03-signal-design

### Signals
Custom signals included:
- error_rate (errors divided by requests)
- average latency per request
- throughput
- performance flags based on thresholds

These signals transformed raw system metrics into more meaningful indicators of performance.

### Artifacts
https://github.com/BethSpornitz/cintel-03-signal-design/tree/main/artifacts

The output dataset includes calculated signals and performance classifications.

### Insights
Signal design made it easier to interpret system behavior by combining raw inputs into meaningful metrics, highlighting relationships between errors, latency, and overall performance.

---

## 4. Rolling Monitoring

### Repository Link
https://github.com/BethSpornitz/cintel-04-rolling-monitoring

### Techniques
Rolling windows were used to calculate moving averages over time. This allowed trends to be observed and smoothed out short-term fluctuations in the data.

### Artifacts
https://github.com/BethSpornitz/cintel-04-rolling-monitoring/tree/main/artifacts

The output includes rolling averages for system metrics and healthcare wait time data, along with visualizations of trends over time.

### Insights
Rolling monitoring revealed patterns such as gradual increases in wait times and changes in satisfaction scores, which are not easily visible in static datasets.

---

## 5. Drift Detection

### Repository Link
https://github.com/BethSpornitz/cintel-05-drift-detection

### Techniques
Drift detection was performed by comparing baseline (reference) metrics to current metrics. Differences between these periods were calculated and evaluated against thresholds to determine whether drift occurred.

### Artifacts
https://github.com/BethSpornitz/cintel-05-drift-detection/tree/main/artifacts

The output includes summary tables showing baseline values, current values, and drift indicators for key metrics.

### Insights
The analysis showed increases in requests, errors, and latency, indicating system drift. This helps identify when system performance is changing and may require investigation or corrective action.

---

## 6. Continuous Intelligence Pipeline

### Repository Link
https://github.com/BethSpornitz/cintel-06-continuous-intelligence

### Techniques
This module combined signal design, anomaly detection, and rolling monitoring into a single pipeline. The system calculates rolling metrics, detects anomalies, and assigns an overall system state (Stable, Warning, or Critical).

### Artifacts
https://github.com/BethSpornitz/cintel-06-continuous-intelligence/tree/main/artifacts

The outputs include rolling monitoring datasets, system assessment summaries, and a dashboard visualization.

### Assessment
The pipeline provides an overall system health classification based on multiple signals. In the ED dataset, higher wait times were associated with lower patient satisfaction and increased warning states. This demonstrates how continuous intelligence techniques can support real-time operational decision-making.
