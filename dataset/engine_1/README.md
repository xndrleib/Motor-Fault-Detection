# Engine 1 Data Repository

## Overview
This repository contains experimental time-series data collected from Engine 1 for motor analysis. The data is organized into separate experiment folders (e.g., `experiment_1`, `experiment_2`, etc.) based on the data collection date. Each experiment folder includes the raw data files and an accompanying `experiment_info.yml` file that contains detailed metadata about that experiment. The experiments were conducted on a motor with a configuration specified in `engine.yml`.

## Directory Structure
The repository is organized as follows:

```
engine_1/
├── experiment_1/         # Data collected on 11 January 2023
│   ├── ВКЛ-ВЫКЛ нагрузка 100% 11 01.txt
│   ├── нагрузка 100% 11 01.txt
│   ├── холостой ход 11 01.txt
│   └── experiment_info.yml
├── experiment_2/         # Data collected on 20 January 2023
│   ├── 20_01_1460_об_мин_с_дисбалансом_ротора.txt
│   └── experiment_info.yml
├── experiment_3/         # Data collected on 7 February 2023
│   ├── 0702  межвитковые фазы A ток фазы А.txt
│   ├── 0702  перекос фазы A ток фазы А.txt
│   ├── 0702 без перекосов фаза A.txt
│   └── experiment_info.yml
├── experiment_4/         # Data collected on 21 February 2023
│   ├── 21 02 дисбаланс макс обороты 100%.txt
│   └── experiment_info.yml
└── README.md             # This file
```

## Data File Descriptions

### Experiment 1 (11 January 2023)
- **Files:**
  1. **ВКЛ-ВЫКЛ нагрузка 100% 11 01.txt**  
     *Description:* Data of motor operation under 100% load with switching on/off.  
     *Experiment Time:* 11 January 2023, 12:05:10  
     *Module:* E-2010B (8T773102)  
     *Number of Frames:* 600,000  
     *Input Rate:* 10 kHz  
     *Channel:* 2  
     *Time Markers Scale:* Seconds

  2. **нагрузка 100% 11 01.txt**  
     *Description:* Data of motor operating at 100% load.  
     *Experiment Time:* 11 January 2023, 12:02:45  
     *Module:* E-2010B (8T773102)  
     *Number of Frames:* 600,000  
     *Input Rate:* 10 kHz  
     *Channel:* 2  
     *Time Markers Scale:* Seconds

  3. **холостой ход 11 01.txt**  
     *Description:* Data from a motor running at no load (idle).  
     *Experiment Time:* 11 January 2023, 11:59:08  
     *Module:* E-2010B (8T773102)  
     *Number of Frames:* 600,000  
     *Input Rate:* 10 kHz  
     *Channel:* 2  
     *Time Markers Scale:* Seconds

### Experiment 2 (20 January 2023)
- **File:**
  - **20_01_1460_об_мин_с_дисбалансом_ротора.txt**  
    *Description:* Data from a motor running at 1460 RPM with rotor imbalance.  
    *Experiment Time:* 20 January 2023, 12:49:36  
    *Module:* E-2010B (8T773102)  
    *Number of Frames:* 100,000  
    *Input Rate:* 10 kHz  
    *Channel:* 2  
    *Time Markers Scale:* Seconds

### Experiment 3 (7 February 2023)
- **Files:**
  1. **0702  межвитковые фазы A ток фазы А.txt**  
     *Description:* Data related to phase A currents with inter-turn faults.  
     *Experiment Time:* 7 February 2023, 11:11:56  
     *Module:* E-2010B (8T773112)  
     *Number of Frames:* 50,000  
     *Input Rate:* 10 kHz  
     *Channel:* 1  
     *Time Markers Scale:* Milliseconds

  2. **0702  перекос фазы A ток фазы А.txt**  
     *Description:* Data from phase A currents with phase imbalance.  
     *Experiment Time:* 7 February 2023, 10:49:31  
     *Module:* E-2010B (8T773112)  
     *Number of Frames:* 50,000  
     *Input Rate:* 10 kHz  
     *Channel:* 1  
     *Time Markers Scale:* Milliseconds

  3. **0702 без перекосов фаза A.txt**  
     *Description:* Data from phase A currents without any imbalances.  
     *Experiment Time:* 7 February 2023, 10:21:48  
     *Module:* E-2010B (8T773112)  
     *Number of Frames:* 100,000  
     *Input Rate:* 10 kHz  
     *Channel:* 1  
     *Time Markers Scale:* Milliseconds

### Experiment 4 (21 February 2023)
- **File:**
  - **21 02 дисбаланс макс обороты 100%.txt**  
    *Description:* Data from a motor operating at maximum RPM with imbalance.  
    *Experiment Time:* 21 February 2023, 10:04:13  
    *Module:* E-2010B (8T773112)  
    *Number of Frames:* 50,000  
    *Input Rate:* 10 kHz  
    *Channel:* 1  
    *Time Markers Scale:* Seconds

## Additional Notes
- Data was acquired using module **E-2010B** with different module IDs as indicated in the file descriptions.
- Although the data is said to have been collected on the LIMAN stand ([LIMAN Stand Details](https://wiki.liman-tech.ru/s/01394367-d50f-4e9c-909c-22c75ce7e308)), the configuration corresponds to a different motor.
- For additional experiment-specific metadata, please refer to the `experiment_info.yml` file in each experiment folder.
