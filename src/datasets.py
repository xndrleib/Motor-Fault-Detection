# datasets.py
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder
from src.anomaly_injector import CompositeAnomalyInjector, NoiseInjector
from src.data_pipeline import time_to_freq_transform
from src.noise_policy import NoisePolicy
from src.normalization import Normalizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from pathlib import Path


def _min_per_class(group: Dict[str, np.ndarray]) -> int:
    """Returns the smallest class count in the group for balancing."""
    return min(len(v) for v in group.values())


def create_balanced_datasets(
    segments: np.ndarray,
    seg_meta_df,
    freqs: np.ndarray,
    fault_freqs: Dict[str, np.ndarray],
    mode: str = "binary",
    normalizer: Optional[Normalizer] = None,
    test_size: float = 0.30,
    val_size: float = 0.20,
    seed: int = 42,
    anomaly_injector: Optional[Any] = None,
    real_fault_train: int = 0,
    save_indices: bool = False,
    indices_dir: Optional[os.PathLike] = None,
    return_indices: bool = False,
    train_dataset_cls=None,
    train_dataset_kwargs: Optional[dict] = None,
    time_segments: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Split → balance → build (train, val, test) datasets.

    *If ``normalizer`` is global and not yet fitted, it is fitted on all
    **training** windows before dataset construction.*

    Accepts a custom Dataset class (train_dataset_cls) for training set, and kwargs for its configuration.
    Optionally accepts ``time_segments`` aligned with ``segments`` for time-domain augmentation.
    """
    if train_dataset_cls is None:
        train_dataset_cls = HybridAugFaultDataset
    if train_dataset_kwargs is None:
        train_dataset_kwargs = {}

    rng = np.random.default_rng(seed)

    # ── 1. stratified index split (unchanged) ──────────────────────────────
    normal_idx = seg_meta_df.index[seg_meta_df["state"] == "normal"].to_numpy()
    fault_idx_all = seg_meta_df.index[seg_meta_df["state"] != "normal"].to_numpy()

    real_fault_train = min(int(real_fault_train), len(fault_idx_all))
    train_fault_idx = rng.choice(fault_idx_all, real_fault_train, replace=False)
    fault_idx_eval = np.setdiff1d(fault_idx_all, train_fault_idx)

    norm_train_idx, norm_test_idx = train_test_split(
        normal_idx, test_size=test_size, random_state=seed, shuffle=True
    )
    norm_train_idx, norm_val_idx = train_test_split(
        norm_train_idx,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        shuffle=True,
    )

    fault_val_idx, fault_test_idx = train_test_split(
        fault_idx_eval,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        shuffle=True,
    )

    # ── 2. undersample VAL / TEST for perfect balance ─────────────────────
    if mode == "binary":
        by_cls_val = {"normal": norm_val_idx, "fault": fault_val_idx}
        by_cls_test = {"normal": norm_test_idx, "fault": fault_test_idx}
        n_val = _min_per_class(by_cls_val)
        n_test = _min_per_class(by_cls_test)
        val_idx = np.concatenate(
            [
                rng.choice(by_cls_val["normal"], n_val, replace=False),
                rng.choice(by_cls_val["fault"], n_val, replace=False),
            ]
        )
        test_idx = np.concatenate(
            [
                rng.choice(by_cls_test["normal"], n_test, replace=False),
                rng.choice(by_cls_test["fault"], n_test, replace=False),
            ]
        )
    else:  # multiclass
        classes = seg_meta_df["state"].unique().tolist()
        by_cls_val = {
            c: seg_meta_df.index[
                (seg_meta_df["state"] == c)
                & (
                    seg_meta_df.index.isin(norm_val_idx)
                    | seg_meta_df.index.isin(fault_val_idx)
                )
            ].to_numpy()
            for c in classes
        }
        by_cls_test = {
            c: seg_meta_df.index[
                (seg_meta_df["state"] == c)
                & (
                    seg_meta_df.index.isin(norm_test_idx)
                    | seg_meta_df.index.isin(fault_test_idx)
                )
            ].to_numpy()
            for c in classes
        }
        n_val = _min_per_class(by_cls_val)
        n_test = _min_per_class(by_cls_test)
        val_idx = np.concatenate(
            [rng.choice(by_cls_val[c], n_val, False) for c in classes]
        )
        test_idx = np.concatenate(
            [rng.choice(by_cls_test[c], n_test, False) for c in classes]
        )

    train_idx = np.concatenate([norm_train_idx, train_fault_idx])

    # ── 3. persist indices (optional) ─────────────────────────────────────
    if save_indices:
        p = Path(indices_dir or os.getcwd())
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / f"train_idx_{mode}.npy", train_idx)
        np.save(p / f"val_idx_{mode}.npy", val_idx)
        np.save(p / f"test_idx_{mode}.npy", test_idx)
        print(f"[✓] index arrays saved to {p.resolve()}/")

    # ── 4. build NumPy views ──────────────────────────────────────────────
    i2r = seg_meta_df.index.get_indexer
    X_trn = segments[i2r(train_idx)]
    X_val = segments[i2r(val_idx)]
    X_tst = segments[i2r(test_idx)]

    # ── 5. fit global normaliser **once** on TRAIN windows ───────────────
    if (
        normalizer is not None
        and normalizer.mode == "global"
        and normalizer.stats is None
    ):
        normalizer.fit(X_trn)

    # ── 6. normalise VAL / TEST immediately (TRAIN is done on‑the‑fly) ───
    if normalizer is not None:
        X_val = normalizer.transform(X_val)
        X_tst = normalizer.transform(X_tst)

    # ── 7. labels for evaluation tensors ─────────────────────────────────
    y_val_raw = seg_meta_df.loc[val_idx, "state"].values
    y_tst_raw = seg_meta_df.loc[test_idx, "state"].values

    if mode == "binary":
        y_val = (y_val_raw != "normal").astype(np.int64)
        y_tst = (y_tst_raw != "normal").astype(np.int64)
        label_encoder = None
    else:
        label_encoder = LabelEncoder().fit(seg_meta_df["state"])
        y_val = label_encoder.transform(y_val_raw).astype(np.int64)
        y_tst = label_encoder.transform(y_tst_raw).astype(np.int64)

    # ── 9. construct datasets ───────────────────────────────────────────

    time_trn = None
    if time_segments is not None:
        if len(time_segments) != len(segments):
            raise ValueError("time_segments must have the same length as segments.")
        time_trn = time_segments[train_idx]

    train_ds = train_dataset_cls(
        segments=X_trn,
        seg_meta_df=seg_meta_df.loc[train_idx],
        freqs=freqs,
        fault_freqs=fault_freqs,
        mode=mode,
        normalizer=normalizer,
        anomaly_injector=anomaly_injector,
        time_segments=time_trn,
        **train_dataset_kwargs,
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_tst, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_tst, dtype=torch.long),
    )

    out: Dict[str, Any] = {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "normalizer": normalizer,
    }
    if mode == "multiclass":
        out["label_encoder"] = label_encoder
    if return_indices:
        out.update({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx})
    return out


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
        self.segments = segments
        self.meta = seg_meta_df.reset_index(drop=True)  # align i = row
        self.freqs = freqs
        self.fault_freqs = fault_freqs
        self.fault_types = sorted(fault_freqs.keys())
        self.mode = mode
        self.norm = normalizer
        self.inj = anomaly_injector
        self.p = p_inject
        self.cache_flag = cache

        # RNG - worker-safe (DataLoader forks after construction)
        self.rng = np.random.RandomState(seed)

        # LabelEncoder only if needed
        if mode == "multiclass":
            self.le = LabelEncoder().fit(self.fault_types + ["normal"])
        else:
            self.le = None

        # maps built in .refresh()
        self.flip_mask = None  # bool (N,)   - which normals flip
        self.assigned_fault = None  # str  (N,)   - fault type per flip
        self._cached_xy = None  # list[(Tensor,Tensor)] if cache==True

        self.refresh(0)  # build epoch-0 state

    def refresh(self, epoch_seed: int | None = None) -> None:
        """Redraw which normal windows turn into which synthetic faults."""
        if epoch_seed is not None:
            self.rng.seed(epoch_seed)

        is_normal = self.meta["state"].to_numpy() == "normal"
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
                    segment=seg,
                    fft_freqs=self.freqs,
                    fault_freqs=self.fault_freqs[ftype],
                    injector_keys=["peak-anomaly"],
                )
            chosen_state = ftype
        else:
            chosen_state = orig_state

        # 2) optional noise injection (always last)
        if self.inj is not None and "noise" in self.inj.injectors:
            seg = self.inj.inject(
                segment=seg,
                fft_freqs=self.freqs,
                fault_freqs=None,
                injector_keys=["noise"],
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
        self.mode = mode
        self.norm = normalizer
        self.freqs = freqs
        self.inj = injector
        self.rng = np.random.RandomState(rng_seed)

        # 1. separate original arrays
        is_norm = seg_meta_df["state"].to_numpy() == "normal"
        self.orig_norm = segments[is_norm]  # (N, L)
        self.orig_meta = seg_meta_df[is_norm].reset_index()  # keep for src idx
        N, L = self.orig_norm.shape
        self.fault_types = sorted(fault_freqs.keys())

        # 2. allocate augmented arrays ------------------------------------------------
        num_fault = N * K * len(self.fault_types)
        num_noisy = N * R
        num_orig = N  # keep originals
        total = num_fault + num_noisy + num_orig

        self.pool = np.empty((total, L), dtype=np.float32)
        self.cls = np.empty(total, dtype=object)  # class names
        self.src = np.empty(total, dtype=object)  # 'orig'/'noise'/'gauss'

        write_ptr = 0

        # 2-a. original normals (always included)
        self.pool[write_ptr : write_ptr + N] = self.orig_norm
        self.cls[write_ptr : write_ptr + N] = "normal"
        self.src[write_ptr : write_ptr + N] = "orig"
        write_ptr += N

        # 2-b. R noisy-normal variants
        if R > 0:
            noise_inj = NoiseInjector(noise_factor=0.05)
            for r in range(R):
                noisy = np.stack([noise_inj.inject(s) for s in self.orig_norm])
                self.pool[write_ptr : write_ptr + N] = noisy
                self.cls[write_ptr : write_ptr + N] = "normal"
                self.src[write_ptr : write_ptr + N] = f"noise{r}"
                write_ptr += N

        # 2-c. K fault variants for every fault type
        if K > 0:
            for f in self.fault_types:
                ffreq = fault_freqs[f]
                for k in range(K):
                    aug = np.stack(
                        [
                            injector.inject(
                                segment=s,
                                fft_freqs=freqs,
                                fault_freqs=ffreq,
                                injector_keys=["peak-anomaly"],
                            )
                            for s in self.orig_norm
                        ]
                    )
                    self.pool[write_ptr : write_ptr + N] = aug
                    self.cls[write_ptr : write_ptr + N] = f
                    self.src[write_ptr : write_ptr + N] = f"gauss{k}"
                    write_ptr += N

        assert write_ptr == total, "allocation mismatch"

        # 3. build class-to-int mapping
        if mode == "binary":
            # 'normal'=0, 'fault'=1
            self.y_int = np.where(self.cls == "normal", 0, 1).astype(np.int64)
            self.le = None
        else:
            self.le = LabelEncoder().fit(np.unique(self.cls))
            self.y_int = self.le.transform(self.cls)

        # 4. shuffle once (optional, for contiguous class blocks)
        perm = self.rng.permutation(total)
        self.pool = self.pool[perm]
        self.y_int = self.y_int[perm]

    def __len__(self) -> int:
        return self.pool.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = self.pool[idx].copy()  # (L,)
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
    K : int   - fault variants per fault-type & window  (default 2)
    R : int   - noisy-normal variants per window        (default 1)
    keep_orig : bool  - store an untouched copy of every normal window

    Time-domain noise augmentation (optional)
    -----------------------------------------
    time_segments : np.ndarray | None
        Raw time-domain segments aligned with ``segments``. Required if
        ``noise_policy`` is provided.
    noise_policy : NoisePolicy | None
        Sampling policy for additive noise in the time domain.
    noise_total_epochs : int | None
        Total number of epochs (used for curriculum scheduling).
    noise_fft_params : dict | None
        FFT parameters: ``f_sampling``, ``cutoff_freq``, ``db``.
    noise_rng_seed : int | None
        RNG seed for time-domain noise sampling.
    """

    def __init__(
        self,
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
        cache: bool = False,
        time_segments: Optional[np.ndarray] = None,
        noise_policy: Optional[NoisePolicy] = None,
        noise_total_epochs: Optional[int] = None,
        noise_fft_params: Optional[Dict[str, object]] = None,
        noise_rng_seed: Optional[int] = None,
    ):
        super().__init__()
        self.segments = segments
        self.meta = seg_meta_df.reset_index(drop=True)
        self.freqs = freqs
        self.fault_f = fault_freqs
        self.fault_t = sorted(fault_freqs.keys())
        self.mode = mode
        self.norm = normalizer
        self.inj = anomaly_injector
        self.K, self.R = int(K), int(R)
        self.keep_orig = keep_orig
        self.cache_on = cache
        self.rng = np.random.RandomState(seed)
        self.time_segments = time_segments
        self.noise_policy = noise_policy
        self.noise_total_epochs = noise_total_epochs
        self.noise_epoch = 0
        self.noise_rng = np.random.default_rng(
            noise_rng_seed if noise_rng_seed is not None else (seed + 12345)
        )
        self.noise_fft_params = noise_fft_params or {}

        if self.noise_policy is not None and self.time_segments is None:
            raise ValueError("time_segments must be provided when noise_policy is set.")
        if self.time_segments is not None and len(self.time_segments) != len(self.segments):
            raise ValueError("time_segments must align with segments length.")
        if self.noise_policy is not None:
            for key in ("f_sampling", "cutoff_freq", "db"):
                if key not in self.noise_fft_params:
                    raise ValueError(
                        f"noise_fft_params must include '{key}' when noise_policy is set."
                    )

        if self.inj is None and (self.K > 0 or self.R > 0):
            raise ValueError("K>0 or R>0 require a non-None anomaly_injector.")

        # label encoder
        if mode == "multiclass":
            self.le = LabelEncoder().fit(self.fault_t + ["normal"])
        else:
            self.le = None

        self.refresh(0)  # build epoch-0

    # ────────────────────────────────────────────────────────────────────
    def refresh(self, epoch_seed: int | None = None) -> None:
        if epoch_seed is not None:
            self.rng.seed(epoch_seed)
            self.noise_epoch = int(epoch_seed)

        self.variant_map: List[Tuple[int, str, str, int]] = []
        # tuple = (base_idx, base_state, variant_tag, fault_idx)

        GAUSS = "__gauss"
        for idx, state in enumerate(self.meta["state"]):
            if state == "normal":
                # untouched copy
                if self.keep_orig:
                    self.variant_map.append((idx, "normal", "orig", -1))

                # R noisy normals
                for r in range(self.R):
                    self.variant_map.append((idx, "normal", f"noise{r}", -1))

                # K synthetic faults
                for k in range(self.K):
                    f_idx = self.rng.randint(len(self.fault_t))
                    ftype = self.fault_t[f_idx]
                    tag = f"{ftype}{GAUSS}{k}"
                    self.variant_map.append((idx, ftype, tag, f_idx))
            else:
                # real fault window - keep single entry
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

        if self.time_segments is not None and self.noise_policy is not None:
            time_seg = self.time_segments[idx].copy()
            time_seg, _ = self.noise_policy.apply(
                time_seg,
                rng=self.noise_rng,
                epoch=self.noise_epoch,
                total_epochs=self.noise_total_epochs,
            )
            seg, _ = time_to_freq_transform(
                time_seg,
                f_sampling=self.noise_fft_params["f_sampling"],
                cutoff_freq=self.noise_fft_params["cutoff_freq"],
                db=self.noise_fft_params["db"],
            )

        # --- decide variant --------------------------------------------------
        if tag.startswith("noise"):
            seg = self.inj.inject(
                seg, self.freqs, fault_freqs=None, injector_keys=["noise"]
            )
        elif "__gauss" in tag:  # synthetic Gaussian fault
            ftype = self.fault_t[f_idx]
            seg = self.inj.inject(
                segment=seg,
                fft_freqs=self.freqs,
                fault_freqs=self.fault_f[ftype],
                injector_keys=["peak-anomaly"],
            )
            base_state = ftype  # label as fault

            # optional additive noise **after** peaks
            if "noise" in self.inj.injectors:
                seg = self.inj.inject(
                    seg, self.freqs, fault_freqs=None, injector_keys=["noise"]
                )

        # else: 'orig' or 'real' - keep as is

        # --- normalisation ---------------------------------------------------
        if self.norm is not None:
            seg = self.norm.transform(seg[None, ...])[0]

        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)

        if self.mode == "binary":
            y = torch.tensor(0 if base_state == "normal" else 1, dtype=torch.long)
        else:
            y = torch.tensor(int(self.le.transform([base_state])[0]), dtype=torch.long)
        return x, y
