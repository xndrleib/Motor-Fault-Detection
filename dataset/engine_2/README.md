# Defective Motor Dataset – Load Variation Experiments

## Overview
This repository contains time-series measurements (both current and vibration) collected from a set of identical motors operated under varying load conditions. The motors were not controlled by speed (which remains constant) – instead, load variations were introduced via generator parameter adjustments. Data were acquired on **03.10.2024** on the **[Gray Stand](https://wiki.liman-tech.ru/s/e322948b-9045-4d05-99ae-bcb9f495a847)**.

## Motor Defects
The dataset covers four motors with the following defect conditions:
- **Engine #1:** Rotor cage break
- **Engine #2:** No faults
- **Engine #3:** Bearing defect
- **Engine #4:** Stator inter-turn short-circuit

Each motor’s dataset is organized in a similar structure, with subfolders for different load levels.

## Data Format and Folder Structure
All data are stored in CSV format. The repository is organized into two main measurement types:
- **current/** – Contains time-series current measurements.
- **vibration/** – Contains time-series vibration measurements.

Within each of these folders, the data are further subdivided into load condition folders. For example, under the `current/` directory, you might see:
```
experiment_1/current/1st_load_0
experiment_1/current/1st_load_20
experiment_1/current/1st_load_40
experiment_1/current/1st_load_60
experiment_1/current/1st_load_80
experiment_1/current/1st_load_100
experiment_1/current/2nd_load_0
experiment_1/current/2nd_load_20
...
experiment_1/current/4th_load_60
experiment_1/current/4th_load_80
experiment_1/current/4th_load_100
```

### Current Data Details
- **Subfolder Structure:**  
  Each load folder under `current/` is further subdivided by motor phase (typically 1, 2, and 3). For example:
  ```
  experiment_1/current/1st_load_0/1
  experiment_1/current/1st_load_0/2
  experiment_1/current/1st_load_0/3
  ```
- **Content:**  
  Inside each phase folder, multiple CSV files represent individual measurement recordings.

### Vibration Data Details
- **Subfolder Structure:**  
  Under `vibration/`, the load condition folders (e.g., `1st_load_0`, `1st_load_20`, etc.) directly contain CSV files.
- **Content:**  
  Each CSV file corresponds to a single vibration measurement (often identified by a numeric timestamp or sequence number).

## Collection Details
- **Collection Date:** 03.10.2024  
- **Stand:** [Gray Stand](https://wiki.liman-tech.ru/s/e322948b-9045-4d05-99ae-bcb9f495a847)  
- **Data Format:** CSV

## Engine Configuration
All motors in this dataset share the same configuration specified in `engine.yml`.

## Experiment Metadata
Each experiment folder (e.g., `experiment_1`) includes an `experiment_info.yml` file with detailed metadata. This file documents the load conditions, measurement types, defect present, and other experimental notes.

## Additional Notes
- **Data Acquisition:**  
  The current and vibration measurements were not collected concurrently, even though they were obtained using the same equipment.
- **Possible Problem with Phase 3:**
  The very first value in the current time series for phase 3 is missing.
- **Consistency Across Motors:**  
  While the motors are identical in design and configuration, each has been subjected to different defect conditions to study the impact under varying load levels.
- **Usage:**  
  This dataset is intended for researchers and engineers performing fault diagnosis, condition monitoring, and predictive maintenance analyses on electric motors.
- **Further Documentation:**  
  Refer to the `experiment_info.yml` files within each experiment folder for detailed metadata on each load condition.
