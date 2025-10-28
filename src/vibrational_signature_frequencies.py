# vibrational_signature_frequencies.py
import math
from typing import List, Dict

def get_type1_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate harmonic frequencies of rotor rotation.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f_r' (float): Rotational frequency of the rotor.

    Returns:
        list: Frequencies related to rotor rotation.
    """
    return [engine_config['f_r']]


def get_type2_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate harmonic frequencies in the axial direction.

    Args:
        engine_config (dict): A dictionary containing engine parameters.

    Returns:
        list: Frequencies related to axial direction harmonics (currently not implemented).
    """
    return []  # TODO: Implement this function.


def get_type3_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate frequencies for rotor bar cracks or fractures.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f_r' (float): Rotational frequency of the rotor.
            - 's' (float): Slip of the motor.
            - 'p' (int): Number of pole pairs.

    Returns:
        list: Frequencies related to cracks or fractures in rotor bars.
    """
    f_r = engine_config['f_r']
    s = engine_config['s']
    p = engine_config['p']
    d = f_r * s / p  # Slip frequency
    return [f_r - 2 * p * d, f_r + 2 * p * d]


def get_type4_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate sideband frequencies around rotor slot harmonics.

    Args:
        engine_config (dict): A dictionary containing engine parameters.

    Returns:
        list: Frequencies for rotor slot harmonics (currently not implemented).
    """
    return []  # TODO: Implement this function.


def get_type5_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate frequencies for imbalance in power supply voltages.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f1' (float): Supply frequency.

    Returns:
        list: Frequencies related to power supply imbalance.
    """
    f1 = engine_config['f1']
    return [2 * f1 - f1 / 3, 2 * f1 + f1 / 3]


def get_type6_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate harmonic frequencies of radial vibration.

    Args:
        engine_config (dict): A dictionary containing engine parameters.

    Returns:
        list: Frequencies related to radial vibration (currently not implemented).
    """
    return []  # TODO: Implement this function.


def get_type7_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate the first harmonic of rotor rotation frequency.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f_r' (float): Rotational frequency of the rotor.

    Returns:
        list: Frequencies for the first harmonic of rotor rotation.
    """
    return [engine_config['f_r']]


def get_type8_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate the first harmonic frequency in radial or axial direction.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f_r' (float): Rotational frequency of the rotor.

    Returns:
        list: Frequencies for the first harmonic in a specific direction.
    """
    return [engine_config['f_r']]


def get_type9_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate frequency related to the product of rotor speed and pole number.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f_r' (float): Rotational frequency of the rotor.
            - 'p' (int): Number of pole pairs.

    Returns:
        list: Frequencies related to inter-turn short circuits in salient pole rotors.
    """
    return [engine_config['f_r'] * 2 * engine_config['p']]


def get_type10_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate sideband frequencies around the rotor's pole-related frequency.

    Args:
        engine_config (dict): A dictionary containing engine parameters:
            - 'f_r' (float): Rotational frequency of the rotor.
            - 'p' (int): Number of pole pairs.

    Returns:
        list: Frequencies related to loosening of salient pole rotor poles.
    """
    return [engine_config['f_r'] * (2 * engine_config['p'] - 1), engine_config['f_r'] * (2 * engine_config['p'] + 1)]


def get_type11_freqs(engine_config: Dict[str, float]) -> List[float]:
    """
    Calculate frequencies related to DC motor commutation issues.

    Args:
        engine_config (dict): A dictionary containing engine parameters.

    Returns:
        list: Frequencies for commutation defects (currently not implemented).
    """
    return []  # TODO: Implement this function.


ANOMALY_FREQS = {
    "Shaft misalignment": get_type1_freqs,
    "Rotor offset from magnetic center": get_type2_freqs,
    "Cracks or fractures in rotor bars": get_type3_freqs,
    "Loosening of rotor cage bars": get_type4_freqs,
    "Power supply imbalance": get_type5_freqs,
    "Rotor shaft crack": get_type6_freqs,
    "Operating frequency near critical speed": get_type7_freqs,
    "Operating frequency near body resonance": get_type8_freqs,
    "Inter-turn short circuits in salient pole rotor": get_type9_freqs,
    "Loosening of salient pole rotor poles": get_type10_freqs,
    "Breaks in DC motor excitation winding": get_type11_freqs,
}
