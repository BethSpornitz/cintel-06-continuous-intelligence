# Apply Ideas

Technique: Combine signals and observations
to evaluate the overall state of a system.

The goal is not only to measure signals,
but to **interpret them** and describe the system's condition.

Earlier modules focused on individual techniques
such as anomaly detection or signal design.
We can bring techniques together to produce a **simple system assessment**.

## Choosing a Dataset

Good datasets for this module typically:

- represent a system with **multiple metrics**
- allow several signals to be interpreted **together**
- provide observations that help explain system behavior

The goal is to summarize **system health**, not just compute individual metrics.

If you would like to apply these skills to a real dataset
instead of the provided example data, see suggested datasets:

https://denisecase.github.io/pro-analytics-02/reference/datasets/cintel/


## Example Systems

### Web Service Monitoring

Possible fields:

- requests
- errors
- latency_ms

Questions to explore:

- What signals indicate system health?
- When do multiple signals indicate system degradation?

### Energy Grid Monitoring

Possible fields:

- demand
- generation
- reserve_capacity

Questions to explore:

- Does the system appear stable or stressed?
- Which signals best indicate system reliability?

### Public Transportation Operations

Possible fields:

- rides
- delays
- cancellations

Questions to explore:

- Do multiple indicators suggest service problems?
- What signals best summarize operational performance?

### Manufacturing Production

Possible fields:

- units_produced
- defects
- downtime_minutes

Questions to explore:

- What signals indicate production health?
- When do multiple metrics suggest system degradation?

### Online Platform Activity

Possible fields:

- active_users
- posts
- error_rate

Questions to explore:

- Is the system stable, growing, or declining?
- Which signals best summarize platform activity?
