# anomaly_injector.py

from abc import ABC, abstractmethod
import numpy as np
import random
from src.electrical_signature_frequencies import ANOMALY_FREQS

class BaseAnomalyInjector(ABC):
    @abstractmethod
    def inject(self, segment, fft_freqs, fault_type=None):
        """
        Injects an anomaly into the provided segment.
        
        Parameters:
            segment (np.ndarray): The FFT segment to modify.
            fft_freqs (np.ndarray): The corresponding frequency bins.
            fault_type (str, optional): The fault type for injection.
        
        Returns:
            np.ndarray: The modified segment with anomaly injected.
        """
        pass

class GaussianPeakInjector(BaseAnomalyInjector):
    def __init__(self, engine_config, peak_segment=6, amplitude_range=(1.0, 5.0), sigma_range=(0.5, 1.5)):
        """
        Parameters:
            engine_config (dict): Engine configuration for fault frequency calculation.
            peak_segment (int): Half-window size (in frequency bins) around each fault frequency.
            amplitude_range (tuple): Range for random peak amplitudes specific to Gaussian peaks.
            sigma_range (tuple): Range to randomly choose sigma for each injection.
        """
        self.engine_config = engine_config
        self.peak_segment = peak_segment
        self.amplitude_range = amplitude_range
        self.sigma_range = sigma_range

    def _generate_gaussian_peak(self, length, amplitude, sigma):
        x = np.arange(length)
        center = length // 2
        profile = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        return profile

    def inject(self, segment, fft_freqs, fault_type):
        modified_segment = segment.copy()
        # Retrieve target fault frequencies using the fault signature function
        fault_freqs = ANOMALY_FREQS.get(fault_type, lambda ec: [])(self.engine_config)
        for freq in fault_freqs:
            amp_val = random.uniform(*self.amplitude_range)
            sigma_val = random.uniform(*self.sigma_range)
            idx = np.argmin(np.abs(fft_freqs - freq))
            start_idx = max(0, idx - self.peak_segment)
            end_idx = min(len(modified_segment), idx + self.peak_segment + 1)
            window_length = end_idx - start_idx
            profile = self._generate_gaussian_peak(window_length, amp_val, sigma_val)
            modified_segment[start_idx:end_idx] += profile
        return modified_segment

class NoiseInjector(BaseAnomalyInjector):
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor

    def inject(self, segment, fft_freqs=None, fault_type=None):
        noise_std = self.noise_factor * np.std(segment)
        noise = np.random.normal(0, noise_std, segment.shape)
        return segment + noise

class CompositeAnomalyInjector(BaseAnomalyInjector):
    def __init__(self, injectors: dict):
        """
        Combines multiple anomaly injectors.
        
        Parameters:
            injectors (dict): A dictionary mapping string keys (e.g. 'anomaly', 'noise')
                              to BaseAnomalyInjector instances.
        """
        self.injectors = injectors

    def inject(self, segment, fft_freqs, fault_type, injector_keys=None):
        """
        Sequentially applies each selected injector's anomaly into the segment.
        
        Parameters:
            segment (np.ndarray): The FFT segment to modify.
            fft_freqs (np.ndarray): Frequency bins corresponding to the FFT segment.
            fault_type (str): The fault type for injection.
            injector_keys (list, optional): List of keys specifying which injectors to apply.
                                            If None, all injectors are applied.
        
        Returns:
            np.ndarray: The modified segment with combined anomalies.
        
        Raises:
            KeyError: If any key in `injector_keys` is not found in `self.injectors`.
        """
        modified_segment = segment.copy()
        
        # Determine which injectors to apply: use all if no keys provided.
        if injector_keys is None:
            keys_to_apply = list(self.injectors.keys())
        else:
            # Validate injector keys
            valid_keys = set(self.injectors.keys())
            invalid_keys = [key for key in injector_keys if key not in valid_keys]
            
            if invalid_keys:
                raise KeyError(
                    f"Invalid injector keys: {invalid_keys}. "
                    f"Available keys: {list(valid_keys)}"
                )

            keys_to_apply = injector_keys

        # Apply injectors
        for key in keys_to_apply:
            injector = self.injectors[key]
            
            try:
                modified_segment = injector.inject(modified_segment, fft_freqs, fault_type)
            except TypeError:
                modified_segment = injector.inject(modified_segment, fft_freqs)

        return modified_segment
