# MCSA

## Parameters

- **$f_s$ (Supply Frequency):**  
  - **Type:** Constant  
  - **Faults Used For:** Rotor bar defect, ITSC, mechanical defects, and air-gap eccentricity analysis.  
  - **Description:** The fundamental frequency of the power supply (typically 50 or 60 Hz).  
  - **Determination:** This is usually readily available from the power grid or motor nameplate. Measurement is straightforward with standard frequency meters.

- **$f_r$ (Rotor/Shaft Frequency):**  
  - **Type:** Operational  
  - **Faults Used For:** Air-gap eccentricity, ITSC, bearing defects, and mechanical defects.  
  - **Description:** Represents the actual rotational frequency of the rotor, varying with load and slip.  
  - **Determination:** Requires measurement of rotor speed using tachometers or encoders. While common sensors can provide this value, accurate determination under dynamic conditions may need precise instrumentation and real-time monitoring.

- **$s$ (Slip):**  
  - **Type:** Operational  
  - **Faults Used For:** Rotor bar defect analysis.  
  - **Description:** A unitless value indicating the difference between synchronous speed and the actual rotor speed.  
  - **Determination:** Calculated from the difference between the synchronous speed (derived from $f_s$ and the motor’s pole configuration) and the measured rotor speed. It is relatively easy to compute if both speeds are known, but may require dynamic measurement during varying loads.

- **$R_s$ (Stator/Slot Factor):**  
  - **Type:** Constant  
  - **Faults Used For:** Air-gap eccentricity analysis.  
  - **Description:** A design parameter related to the stator or slot geometry, used to adjust eccentricity-induced frequency calculations.  
  - **Determination:** Typically available from motor design documents. If unknown, it may require detailed geometric analysis or manufacturer data, which can be challenging without proper documentation.

- **$p$ (Number of Pole Pairs):**  
  - **Type:** Constant  
  - **Faults Used For:** Air-gap eccentricity analysis.  
  - **Description:** Represents half the number of poles in the motor, a fixed design parameter.  
  - **Determination:** Easily obtained from the motor’s nameplate or technical specifications provided by the manufacturer.

- **$n$ (Number of Rolling Elements):**  
  - **Type:** Constant  
  - **Faults Used For:** Bearing defect calculations (e.g., outer race, inner race, rolling element defects).  
  - **Description:** The fixed count of rolling elements in the bearing, determined by its design.  
  - **Determination:** Clearly specified in the bearing’s design documentation; if not available, it might require physical inspection which is usually straightforward.

- **$D_{pit}$ (Pitch Diameter):**  
  - **Type:** Constant  
  - **Faults Used For:** Bearing defect frequency calculations.  
  - **Description:** The mean diameter on which the bearing’s rolling elements are arranged.  
  - **Determination:** Normally provided in technical specifications. If missing, precise measurements using calipers or imaging techniques are needed, which can be moderately challenging without access to the bearing.

- **$D_{ball}$ (Rolling Element Diameter):**  
  - **Type:** Constant  
  - **Faults Used For:** Bearing defect frequency calculations.  
  - **Description:** The diameter of each individual rolling element in the bearing.  
  - **Determination:** Available from manufacturer data or design documents. Measurement is relatively simple with proper tools if the data is not provided.

- **$\beta$ (Contact Angle):**  
  - **Type:** Constant  
  - **Faults Used For:** Bearing defect frequency calculations.  
  - **Description:** The contact angle (in radians) at which rolling elements meet the raceways; its cosine is used in the calculation formulas.  
  - **Determination:** Often specified by the bearing manufacturer. If unknown, determining $\beta$ might require advanced measurement techniques (e.g., optical or coordinate measurements), making it one of the more challenging parameters to obtain without detailed technical data.
