import math
from typing import List, Dict


def get_type1_freqs(engine_config: Dict[str, float], n_range: range = range(1, 4)) -> List[float]:
    """
    Calculate frequencies associated with rotor bar defects.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 's' (float): Slip of the motor.
            - 'f1' (float): Supply frequency.
        n_range (range): Range of harmonics to consider.

    Returns:
        list: Frequencies related to rotor bar defects.
    """
    s = engine_config['s']
    f1 = engine_config['f1']
    freqs = []
    for n in n_range:
        assert n > 0, f"Harmonic index n must be positive. Got n={n}."
        assert isinstance(n, int), f"Harmonic index n must be an integer. Got type={type(n)}."
        delta = 2 * n * s
        freqs.extend([(1 + delta) * f1, (1 - delta) * f1])
    return freqs


def get_type2_freqs(engine_config: Dict[str, float], n_range: range = range(1, 4)) -> List[float]:
    """
    Calculate frequencies associated with air-gap eccentricity.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'R_s' (float): Stator radius.
            - 'p' (int): Number of pole pairs.
            - 's' (float): Slip of the motor.
            - 'f1' (float): Supply frequency.
        n_range (range): Range of harmonics to consider.

    Returns:
        list: Frequencies related to air-gap eccentricity.
    """
    R_s = engine_config['R_s']
    p = engine_config['p']
    s = engine_config['s']
    f1 = engine_config['f1']
    freqs = []
    base = R_s * (1 - s) / p
    offset = (1 - s) / p

    for n in n_range:
        assert n > 0, f"Harmonic index n must be positive. Got n={n}."
        assert isinstance(n, int), f"Harmonic index n must be an integer. Got type={type(n)}."
        freqs.extend([f1 * (base + n + offset), 
                      f1 * (base + n - offset), 
                      f1 * (base - n + offset), 
                      f1 * (base - n - offset)])
    return freqs


def get_type3_freqs(engine_config: Dict[str, float], n_range: range = range(1, 4), k_range: range = range(1, 4)) -> List[float]:
    """
    Calculate frequencies associated with inter-turn short circuits.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f1' (float): Supply frequency.
            - 'p' (int): Number of pole pairs.
            - 's' (float): Slip of the motor.
        n_range (range): Range of primary harmonics to consider.
        k_range (range): Range of secondary harmonics to consider.

    Returns:
        list: Frequencies related to inter-turn short circuits.
    """
    f1 = engine_config['f1']
    p = engine_config['p']
    s = engine_config['s']
    freqs = []
    base_factor = (1 - s) / p

    for n in n_range:
        assert n > 0, f"Harmonic index n must be positive. Got n={n}."
        assert isinstance(n, int), f"Harmonic index n must be an integer. Got type={type(n)}."
        for k in k_range:
            assert k > 0, f"Harmonic index k must be positive. Got k={k}."
            assert isinstance(k, int), f"Harmonic index k must be an integer. Got type={type(k)}."
            freqs.extend([f1 * (n * base_factor + k), 
                          f1 * (n * base_factor - k)])
    return freqs


def get_bearing_freqs(engine_config: Dict[str, float], defect_type: str) -> List[float]:
    """
    Calculate frequencies for bearing defects based on defect type.

    Args:
        engine_config (dict): A dictionary containing bearing parameters:
            - 'D_pit' (float): Pitch diameter.
            - 'D_ball' (float): Ball diameter.
            - 'f_r' (float): Rotational frequency.
            - 'beta' (float): Contact angle in radians.
            - 'n' (int): Number of rolling elements (only required for outer/inner race).
        defect_type (str): Type of defect. Options are:
            - 'rolling_element'
            - 'outer_race'
            - 'inner_race'

    Returns:
        list: Frequencies related to the specified bearing defect.
    """
    D_pit = engine_config['D_pit']
    D_ball = engine_config['D_ball']
    f_r = engine_config['f_r']
    beta = engine_config['beta']

    if defect_type == 'rolling_element':
        return [(D_pit / D_ball) * f_r * (1 - (D_ball / (D_pit * math.cos(beta)))**2)]
    elif defect_type in {'outer_race', 'inner_race'}:
        n = engine_config['n']
        assert isinstance(n, int) and n > 0, f"Number of rolling elements must be a positive integer. Got n={n}."
        multiplier = -1 if defect_type == 'outer_race' else 1
        return [(n / 2) * f_r * (1 + multiplier * D_ball / (D_pit * math.cos(beta)))]
    else:
        raise ValueError(f"Invalid defect type: '{defect_type}'. Must be one of 'rolling_element', 'outer_race', 'inner_race'.")


def get_type5_freqs(engine_config: Dict[str, float], n_range: range = range(1, 4)) -> List[float]:
    """
    Calculate frequencies associated with other mechanical defects.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f_r' (float): Rotational frequency.
            - 'f1' (float): Supply frequency.
        n_range (range): Range of harmonics to consider.

    Returns:
        list: Frequencies related to mechanical defects.
    """
    f_r = engine_config['f_r']
    f1 = engine_config['f1']
    freqs = []
    for n in n_range:
        assert n > 0, f"Harmonic index n must be positive. Got n={n}."
        assert isinstance(n, int), f"Harmonic index n must be an integer. Got type={type(n)}."
        freqs.append(f1 + n * f_r)
    return freqs


ANOMALY_FREQS = {
    'rotor bar defect': get_type1_freqs,
    'air-gap eccentricity': get_type2_freqs,
    'inter-turn short circuits': get_type3_freqs,
    'bearing defect (rolling element)': lambda ec: get_bearing_freqs(ec, 'rolling_element'),
    'bearing defect (outer race)': lambda ec: get_bearing_freqs(ec, 'outer_race'),
    'bearing defect (inner race)': lambda ec: get_bearing_freqs(ec, 'inner_race'),
    'other mechanical defects': get_type5_freqs
}
