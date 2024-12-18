import math
from decimal import Decimal, getcontext
from typing import Dict, List

# Set a higher precision for Decimal calculations
getcontext().prec = 50

def safe_cos(beta: Decimal) -> Decimal:
    # Compute cosine using Decimal's local context
    # Convert beta (Decimal) to float for math.cos,
    # then back to Decimal for consistency
    val = Decimal(math.cos(float(beta)))
    return val

def get_rotor_bar_freqs(engine_config: Dict[str, float], n_range=range(1,4)) -> List[float]:
    """
    Compute rotor bar defect frequencies for an induction motor.
    
    Formula:
        f = f1 * (1 ± 2 * n * s)
    
    Parameters
    ----------
    engine_config : dict
        'f1': Supply frequency (Hz)
        's' : Slip (unitless)
    n_range : range, optional
    
    Returns
    -------
    List[float]
    
    References
    ----------
    [1] Thomson & Culbert (2014)
    [2] Bellini et al. (2008)
    """
    f1 = Decimal(str(engine_config['f1']))
    s = Decimal(str(engine_config['s']))
    
    freqs = []
    for n in n_range:
        n_d = Decimal(n)
        delta = Decimal('2') * n_d * s
        f_plus = f1 * (Decimal('1') + delta)
        f_minus = f1 * (Decimal('1') - delta)
        freqs.extend([f_plus, f_minus])
    # Convert back to float for final output
    return sorted(float(f) for f in freqs)

def get_eccentricity_freqs(engine_config: Dict[str, float], n_range=range(1,4), method='slot-based') -> List[float]:
    """
    Compute air-gap eccentricity frequencies.
    
    Methods:
    - 'slot-based': f = |f1 ± n*(R_s/p)*f_r|
    - 'simple': f = f1 ± n*f_r
    
    Parameters
    ----------
    engine_config : dict
        'f1', 'f_r', 'R_s', 'p'
    n_range : range
    method : str
    
    Returns
    -------
    List[float]
    
    Raises
    ------
    ValueError: If p ≈ 0
    """
    f1 = Decimal(str(engine_config['f1']))
    f_r = Decimal(str(engine_config['f_r']))
    R_s = Decimal(str(engine_config['R_s']))
    p = Decimal(str(engine_config['p']))
    
    if p == 0:
        raise ValueError("Number of pole pairs p must not be zero to avoid division by zero.")
    
    freqs = []
    for n in n_range:
        n_d = Decimal(n)
        if method == 'slot-based':
            factor = (R_s/p)*f_r
            f_plus = f1 + n_d*factor
            f_minus = f1 - n_d*factor
        else:
            f_plus = f1 + n_d*f_r
            f_minus = f1 - n_d*f_r
        freqs.extend([f_plus, f_minus])
    return sorted(float(f) for f in freqs)

def get_itsc_freqs(engine_config: Dict[str, float], k_values=range(1,4), m_range=range(1,4)) -> List[float]:
    """
    Compute inter-turn short circuit related frequencies.
    
    Formulae often consider:
    - k*f1 and sidebands (k*f1 ± m*f_r)
    
    Parameters
    ----------
    engine_config : dict
        'f1', 'f_r'
    k_values : range
    m_range : range
    
    Returns
    -------
    List[float]
    """
    f1 = Decimal(str(engine_config['f1']))
    f_r = Decimal(str(engine_config['f_r']))
    freqs = []
    
    for k in k_values:
        k_d = Decimal(k)
        main = k_d * f1
        candidates = [main]
        for m in m_range:
            m_d = Decimal(m)
            candidates.append(main + m_d*f_r)
            candidates.append(main - m_d*f_r)
        freqs.extend(candidates)
    return sorted(float(f) for f in freqs)

def get_bearing_freqs(engine_config: Dict[str, float], defect_type: str) -> List[float]:
    """
    Compute bearing defect frequencies for a given bearing configuration and defect type.
    
    The following standard formulas are used (assuming a rolling element bearing):
    
    Let:
    - n: Number of rolling elements
    - D_pit: Pitch (mean) diameter of the bearing (m)
    - D_ball: Diameter of the rolling element (m)
    - f_r: Shaft (rotational) frequency in Hz
    - beta: Contact angle of the rolling elements (radians)
    
    Defect Types and Formulas:
    --------------------------
    Outer Race (BPFO):
        BPFO = (n/2)*f_r * [1 - (D_ball/D_pit)*cos(beta)]
    
    Inner Race (BPFI):
        BPFI = (n/2)*f_r * [1 + (D_ball/D_pit)*cos(beta)]
    
    Rolling Element (BSF):
        BSF = (D_pit/(2*D_ball))*f_r * [1 - ((D_ball/D_pit)*cos(beta))^2]
    
    Parameters
    ----------
    engine_config : dict
        Contains keys:
            'n': Number of rolling elements (integer)
            'D_pit': Pitch diameter of the bearing (float)
            'D_ball': Rolling element diameter (float)
            'f_r': Shaft frequency in Hz (float)
            'beta': Contact angle in radians (float)
    defect_type : str
        Type of bearing defect. Must be one of:
        'outer_race', 'inner_race', or 'rolling_element'
    
    Returns
    -------
    List[float]
        A single-element list containing the computed defect frequency.
    
    Raises
    ------
    ValueError
        If cos(beta) is zero or invalid, or if an invalid defect_type is provided.
    """
    D_pit = Decimal(str(engine_config['D_pit']))
    D_ball = Decimal(str(engine_config['D_ball']))
    f_r = Decimal(str(engine_config['f_r']))
    beta = Decimal(str(engine_config['beta']))
    n = Decimal(str(engine_config['n']))
    
    c_beta = safe_cos(beta)
    if c_beta == 0 or c_beta.is_nan():
        raise ValueError("cos(beta) is zero or invalid, bearing calculation unstable.")
    
    factor = (D_ball / D_pit) * c_beta

    if defect_type == 'rolling_element':
        # BSF formula
        term = factor**Decimal('2')
        freq = (D_pit/(Decimal('2')*D_ball))*f_r*(Decimal('1') - term)
    elif defect_type == 'outer_race':
        # BPFO formula
        freq = (n/Decimal('2'))*f_r*(Decimal('1') - factor)
    elif defect_type == 'inner_race':
        # BPFI formula
        freq = (n/Decimal('2'))*f_r*(Decimal('1') + factor)
    else:
        raise ValueError("Invalid defect type. Must be 'outer_race', 'inner_race', or 'rolling_element'.")
    
    return [float(freq)]


def get_mech_freqs(engine_config: Dict[str, float], n_range=range(1,4)) -> List[float]:
    """
    Compute mechanical defect frequencies.
    
    f = f1 ± n*f_r
    
    Parameters
    ----------
    engine_config : dict
        'f1', 'f_r'
    n_range : range
    
    Returns
    -------
    List[float]
    """
    f1 = Decimal(str(engine_config['f1']))
    f_r = Decimal(str(engine_config['f_r']))
    
    freqs = []
    for n in n_range:
        n_d = Decimal(n)
        freqs.append(f1 + n_d*f_r)
        freqs.append(f1 - n_d*f_r)
    return sorted(float(f) for f in freqs)

ANOMALY_FREQS = {
    'rotor bar defect': get_rotor_bar_freqs,
    'air-gap eccentricity': get_eccentricity_freqs,
    'inter-turn short circuits': get_itsc_freqs,
    'bearing defect (rolling element)': lambda ec: get_bearing_freqs(ec, 'rolling_element'),
    'bearing defect (outer race)': lambda ec: get_bearing_freqs(ec, 'outer_race'),
    'bearing defect (inner race)': lambda ec: get_bearing_freqs(ec, 'inner_race'),
    'other mechanical defects': get_mech_freqs
}

if __name__ == '__main__':
    import yaml

    with open('engine_configs/LIMAN.yml', 'r') as f:
        engine_config = yaml.safe_load(f)

    print("Rotor bar defect frequencies:", ANOMALY_FREQS['rotor bar defect'](engine_config))
    print("Eccentricity frequencies:", ANOMALY_FREQS['air-gap eccentricity'](engine_config))
    print("Inter-turn short circuit frequencies:", ANOMALY_FREQS['inter-turn short circuits'](engine_config))
    print("Bearing rolling element defect frequency:", ANOMALY_FREQS['bearing defect (rolling element)'](engine_config))
    print("Bearing outer race defect frequency:", ANOMALY_FREQS['bearing defect (outer race)'](engine_config))
    print("Bearing inner race defect frequency:", ANOMALY_FREQS['bearing defect (inner race)'](engine_config))
    print("Other mechanical defect frequencies:", ANOMALY_FREQS['other mechanical defects'](engine_config))


    # Rotor bar defect frequencies: [41.6, 44.4, 47.2, 52.8, 55.6, 58.4]
    # Eccentricity frequencies: [-1139.983, -743.322, -346.661, 446.661, 843.322, 1239.983]
    # Inter-turn short circuit frequencies: [-19.999, 3.334, 26.667, 30.001, 50.0, 53.334, 73.333, 76.667, 80.001, 96.666, 100.0, 103.334, 119.999, 123.333, 126.667, 146.666, 150.0, 169.999, 173.333, 196.666, 219.999]
    # Bearing rolling element defect frequency: [-47.576496009847844]
    # Bearing outer race defect frequency: [-308.93442795570587]
    # Bearing inner race defect frequency: [495.59842795570586]
    # Other mechanical defect frequencies: [-19.999, 3.334, 26.667, 73.333, 96.666, 119.999]