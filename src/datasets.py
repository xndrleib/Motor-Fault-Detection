
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple
from sklearn.preprocessing import LabelEncoder

class FaultInjectionDataset(Dataset):
    """
    One-stop dataset that (1) decides **once per epoch** which normal windows
    turn into synthetic faults, (2) injects Gaussian peaks + optional noise,
    (3) returns stable (x, y) pairs for the whole epoch, and (4) can be
    refreshed between epochs for new mixes.

    Call `.refresh(epoch)` at the start of every epoch to re-draw the map.

    Parameters
    ----------
    segments          : float32 ndarray, shape (N, L)
    seg_meta_df       : DataFrame row-aligned with *segments*
    freqs             : ndarray (L,) - FFT frequency vector
    fault_freqs       : dict[str, ndarray] - char. freqs per fault type
    mode              : "binary" | "multiclass"
    normalizer        : Normalizer | None - pass global/per normaliser or None
    anomaly_injector  : CompositeAnomalyInjector | None
    p_inject          : float ∈ [0,1] - probability a normal → synthetic fault
    seed              : int - RNG seed (reproducible across workers)
    cache             : bool - if True, store modified windows in RAM
    """

    def __init__(
        self,
        segments: np.ndarray,
        seg_meta_df,
        freqs: np.ndarray,
        fault_freqs: Dict[str, np.ndarray],
        mode: str = "binary",
        normalizer: Optional[Any] = None,
        anomaly_injector: Optional[Any] = None,
        p_inject: float = 0.5,
        seed: int = 0,
        cache: bool = False,
    ):
        super().__init__()
        assert 0.0 <= p_inject <= 1.0
        self.segments     = segments
        self.meta         = seg_meta_df.reset_index(drop=True)   # align i = row
        self.freqs        = freqs
        self.fault_freqs  = fault_freqs
        self.fault_types  = sorted(fault_freqs.keys())
        self.mode         = mode
        self.norm         = normalizer
        self.inj          = anomaly_injector
        self.p            = p_inject
        self.cache_flag   = cache

        # RNG - worker-safe (DataLoader forks after construction)
        self.rng  = np.random.RandomState(seed)

        # LabelEncoder only if needed
        if mode == "multiclass":
            self.le = LabelEncoder().fit(self.fault_types + ["normal"])
        else:
            self.le = None

        # maps built in .refresh()
        self.flip_mask      = None      # bool (N,)   - which normals flip
        self.assigned_fault = None      # str  (N,)   - fault type per flip
        self._cached_xy     = None      # list[(Tensor,Tensor)] if cache==True

        self.refresh(0)     # build epoch-0 state

    def refresh(self, epoch_seed: int | None = None) -> None:
        """Redraw which normal windows turn into which synthetic faults."""
        if epoch_seed is not None:
            self.rng.seed(epoch_seed)

        is_normal  = self.meta["state"].to_numpy() == "normal"
        self.flip_mask = np.zeros(len(self), dtype=bool)
        self.flip_mask[is_normal] = self.rng.rand(is_normal.sum()) < self.p

        # assign concrete fault types for every flipped window
        self.assigned_fault = np.array([""] * len(self), dtype=object)
        n_flips = self.flip_mask.sum()
        if n_flips > 0:
            self.assigned_fault[self.flip_mask] = self.rng.choice(
                self.fault_types, size=n_flips
            )

        # (re)build cache if required
        self._cached_xy = None
        if self.cache_flag:
            self._cached_xy = [self._build_sample(i) for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cached_xy is not None:
            return self._cached_xy[idx]
        return self._build_sample(idx)
    
    def _build_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) WITH deterministic synthetic injection."""
        seg = self.segments[idx].copy()
        orig_state = self.meta.at[idx, "state"]

        # 1) maybe convert to synthetic fault
        if self.flip_mask[idx]:
            ftype = self.assigned_fault[idx]
            if self.inj is not None:
                seg = self.inj.inject(
                    segment       = seg,
                    fft_freqs     = self.freqs,
                    fault_freqs   = self.fault_freqs[ftype],
                    injector_keys = ["peak-anomaly"],
                )
            chosen_state = ftype
        else:
            chosen_state = orig_state

        # 2) optional noise injection (always last)
        if self.inj is not None and "noise" in self.inj.injectors:
            seg = self.inj.inject(
                segment       = seg,
                fft_freqs     = self.freqs,
                fault_freqs   = None,
                injector_keys = ["noise"],
            )

        # 3) optional normalisation
        if self.norm is not None:
            seg = self.norm.transform(seg[None, ...])[0]  # keep 1-D

        # 4) pack
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)  # (1, L)

        if self.mode == "binary":
            y = torch.tensor(0 if chosen_state == "normal" else 1, dtype=torch.long)
        else:
            y = torch.tensor(
                int(self.le.transform([chosen_state])[0]), dtype=torch.long
            )
        return x, y