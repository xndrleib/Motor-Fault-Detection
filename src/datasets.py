
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder
from src.anomaly_injector import CompositeAnomalyInjector, NoiseInjector
from src.normalization import Normalizer

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

class AugmentedPoolDataset(Dataset):
    """
    One-time, in-memory augmentation dataset.

    For every *normal* FFT window the constructor creates:
        • K synthetic-fault variants  (GaussianPeakInjector)
        • R noisy-normal  variants   (NoiseInjector)

    Fault and normal classes are therefore *perfectly balanced* **per window**.
    The augmented arrays live in RAM for the whole training session.

    Parameters
    ----------
    segments, seg_meta_df, freqs :   artefacts from preprocessing.
    fault_freqs : Dict[str, np.ndarray]   characteristic freqs per fault type.
    K : int                              number of fault variants *per* type.
    R : int                              number of noisy-normal variants.
    injector : CompositeAnomalyInjector  must expose 'peak-anomaly' and 'noise'.
    mode : 'binary' | 'multiclass'
    normalizer : Normalizer | None       on-the-fly transform (global/per/None).
    rng_seed : int                       reproducible randomness.
    """

    def __init__(
        self,
        segments: np.ndarray,
        seg_meta_df,
        freqs: np.ndarray,
        fault_freqs: Dict[str, np.ndarray],
        K: int = 2,
        R: int = 1,
        injector: Optional[CompositeAnomalyInjector] = None,
        mode: str = "binary",
        normalizer: Optional[Normalizer] = None,
        rng_seed: int = 0,
    ) -> None:
        super().__init__()
        self.mode   = mode
        self.norm   = normalizer
        self.freqs  = freqs
        self.inj    = injector
        self.rng    = np.random.RandomState(rng_seed)

        # 1. separate original arrays
        is_norm = seg_meta_df["state"].to_numpy() == "normal"
        self.orig_norm = segments[is_norm]                   # (N, L)
        self.orig_meta = seg_meta_df[is_norm].reset_index()  # keep for src idx
        N, L = self.orig_norm.shape
        self.fault_types = sorted(fault_freqs.keys())

        # 2. allocate augmented arrays ------------------------------------------------
        num_fault  = N * K * len(self.fault_types)
        num_noisy  = N * R
        num_orig   = N                                     # keep originals
        total      = num_fault + num_noisy + num_orig

        self.pool  = np.empty((total, L), dtype=np.float32)
        self.cls   = np.empty(total, dtype=object)         # class names
        self.src   = np.empty(total, dtype=object)         # 'orig'/'noise'/'gauss'

        write_ptr = 0

        # 2-a. original normals (always included)
        self.pool[write_ptr:write_ptr+N] = self.orig_norm
        self.cls [write_ptr:write_ptr+N] = "normal"
        self.src [write_ptr:write_ptr+N] = "orig"
        write_ptr += N

        # 2-b. R noisy-normal variants
        if R > 0:
            noise_inj = NoiseInjector(noise_factor=0.05)
            for r in range(R):
                noisy = np.stack([noise_inj.inject(s) for s in self.orig_norm])
                self.pool[write_ptr:write_ptr+N] = noisy
                self.cls [write_ptr:write_ptr+N] = "normal"
                self.src [write_ptr:write_ptr+N] = f"noise{r}"
                write_ptr += N

        # 2-c. K fault variants for every fault type
        if K > 0:
            for f in self.fault_types:
                ffreq = fault_freqs[f]
                for k in range(K):
                    aug = np.stack([
                        injector.inject(
                            segment=s,
                            fft_freqs=freqs,
                            fault_freqs=ffreq,
                            injector_keys=["peak-anomaly"]
                        ) for s in self.orig_norm
                    ])
                    self.pool[write_ptr:write_ptr+N]  = aug
                    self.cls [write_ptr:write_ptr+N]  = f
                    self.src [write_ptr:write_ptr+N]  = f"gauss{k}"
                    write_ptr += N

        assert write_ptr == total, "allocation mismatch"

        # 3. build class-to-int mapping
        if mode == "binary":
            # 'normal'=0, 'fault'=1
            self.y_int = np.where(self.cls == "normal", 0, 1).astype(np.int64)
            self.le    = None
        else:
            self.le = LabelEncoder().fit(np.unique(self.cls))
            self.y_int = self.le.transform(self.cls)

        # 4. shuffle once (optional, for contiguous class blocks)
        perm = self.rng.permutation(total)
        self.pool  = self.pool[perm]
        self.y_int = self.y_int[perm]

    def __len__(self) -> int:
        return self.pool.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = self.pool[idx].copy()                      # (L,)
        if self.norm is not None:
            seg = self.norm.transform(seg[None, ...])[0]
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(int(self.y_int[idx]), dtype=torch.long)
        return x, y

class HybridAugFaultDataset(Dataset):
    """
    *Per-epoch* dynamic dataset that, for every NORMAL FFT window, returns:

        • 1 optional untouched copy            (tag 'orig')
        • R noisy-normal variants              (tag 'noiseᵢ')
        • K variants *per fault class*         (tag '<fault>ₖ')

    Real-fault windows are kept as a single entry (tag 'real').

    Call `.refresh(epoch)` to regenerate noise & Gaussian parameters.

    Parameters
    ----------
    All base arguments mirror FaultInjectionDataset:
        segments, seg_meta_df, freqs, fault_freqs, mode, normalizer,
        anomaly_injector, seed, cache

    Extra augmentation knobs
    ------------------------
    K : int   – fault variants per fault-type & window  (default 2)
    R : int   – noisy-normal variants per window        (default 1)
    keep_orig : bool  – store an untouched copy of every normal window
    """

    def __init__(self,
                 segments: np.ndarray,
                 seg_meta_df,
                 freqs: np.ndarray,
                 fault_freqs: Dict[str, np.ndarray],
                 *,
                 mode: str = "binary",
                 normalizer: Optional[Normalizer] = None,
                 anomaly_injector: Optional[CompositeAnomalyInjector] = None,
                 K: int = 2,
                 R: int = 1,
                 keep_orig: bool = True,
                 seed: int = 0,
                 cache: bool = False):
        super().__init__()
        self.segments  = segments
        self.meta      = seg_meta_df.reset_index(drop=True)
        self.freqs     = freqs
        self.fault_f   = fault_freqs
        self.fault_t   = sorted(fault_freqs.keys())
        self.mode      = mode
        self.norm      = normalizer
        self.inj       = anomaly_injector
        self.K, self.R = int(K), int(R)
        self.keep_orig = keep_orig
        self.cache_on  = cache
        self.rng       = np.random.RandomState(seed)

        # label encoder
        if mode == "multiclass":
            self.le = LabelEncoder().fit(self.fault_t + ["normal"])
        else:
            self.le = None

        self.refresh(0)          # build epoch-0

    # ────────────────────────────────────────────────────────────────────
    def refresh(self, epoch_seed: int | None = None) -> None:
        if epoch_seed is not None:
            self.rng.seed(epoch_seed)

        self.variant_map: List[Tuple[int,str,str,int]] = []
        # tuple = (base_idx, base_state, variant_tag, fault_idx)

        for idx, state in enumerate(self.meta["state"]):
            if state == "normal":
                # untouched copy
                if self.keep_orig:
                    self.variant_map.append((idx, "normal", "orig", -1))

                # R noisy normals
                for r in range(self.R):
                    self.variant_map.append((idx, "normal", f"noise{r}", -1))

                # K synthetic faults per fault class
                for f_idx, ftype in enumerate(self.fault_t):
                    for k in range(self.K):
                        self.variant_map.append((idx, ftype, f"{ftype}_{k}", f_idx))
            else:
                # real fault window – keep single entry
                self.variant_map.append((idx, state, "real", -1))

        # shuffle for randomness
        self.rng.shuffle(self.variant_map)

        # drop cache
        self._cached = None
        if self.cache_on:
            self._cached = [self._build_sample(v) for v in self.variant_map]

    # ────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.variant_map)

    def __getitem__(self, i: int):
        if self._cached is not None:
            return self._cached[i]
        return self._build_sample(self.variant_map[i])

    # ────────────────────────────────────────────────────────────────────
    def _build_sample(self, entry) -> Tuple[torch.Tensor, torch.Tensor]:
        idx, base_state, tag, f_idx = entry
        seg = self.segments[idx].copy()

        # --- decide variant --------------------------------------------------
        if tag.startswith("noise"):
            seg = self.inj.inject(seg, self.freqs, fault_freqs=None, injector_keys=["noise"])

        elif "_" in tag:                            # synthetic Gaussian fault
            ftype = self.fault_t[f_idx]
            seg = self.inj.inject(
                segment       = seg,
                fft_freqs     = self.freqs,
                fault_freqs   = self.fault_f[ftype],
                injector_keys = ["peak-anomaly"]
            )
            base_state = ftype                       # label as fault

            # optional additive noise **after** peaks
            if "noise" in self.inj.injectors:
                seg = self.inj.inject(seg, self.freqs, fault_freqs=None, injector_keys=["noise"])

        # else: 'orig' or 'real' – keep as is

        # --- normalisation ---------------------------------------------------
        if self.norm is not None:
            seg = self.norm.transform(seg[None, ...])[0]

        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)

        if self.mode == "binary":
            y = torch.tensor(0 if base_state == "normal" else 1, dtype=torch.long)
        else:
            y = torch.tensor(int(self.le.transform([base_state])[0]), dtype=torch.long)
        return x, y
