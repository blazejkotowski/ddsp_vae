from __future__ import annotations

import pickle
from typing import Optional

import lmdb
import torch
from torch.utils.data import Dataset

class PriorSequenceDataset(Dataset):
    """Loads control sequences for Prior training.

    Preferred format is LMDB (directory ending with .lmdb).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        in_memory: bool = True,
    ):
        if path is None:
            raise ValueError("PriorSequenceDataset requires 'path'")

        self.path = str(path)
        self.in_memory = bool(in_memory)

        self._data: Optional[torch.Tensor] = None

        # LMDB state
        self._env = None
        self._meta: Optional[dict] = None

        self._init_lmdb()

    def _init_lmdb(self) -> None:
        env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
        with env.begin() as txn:
            meta_buf = txn.get(b"metadata")
            if meta_buf is None:
                raise RuntimeError(f"LMDB prior cache missing metadata: {self.path}")
            meta = pickle.loads(meta_buf)

        self._meta = meta
        self.num_sequences = int(meta["num_sequences"])
        self.seq_len = int(meta["seq_len"])
        self._num_controls = int(meta["num_controls"])

        if self.in_memory:
            # materialize to a single tensor (fastest training);
            # format is float32 bytes, shape [seq_len, D]
            data = torch.empty((self.num_sequences, self.seq_len, self._num_controls), dtype=torch.float32)
            with env.begin() as txn:
                for i in range(self.num_sequences):
                    buf = txn.get(f"controls:{i:08d}".encode())
                    if buf is None:
                        raise RuntimeError(f"Missing key controls:{i:08d} in {self.path}")
                    arr = torch.frombuffer(buf, dtype=torch.float32)
                    data[i] = arr.view(self.seq_len, self._num_controls)
            self._data = data
            try:
                self._data.share_memory_()
            except RuntimeError:
                pass
            env.close()
        else:
            self._env = env

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            finally:
                self._env = None

        # no HDF5 resources

    @property
    def num_controls(self):
        if self._data is not None:
            return int(self._data.shape[2])
        if self._meta is not None:
            return int(self._meta["num_controls"])
        raise RuntimeError("PriorSequenceDataset internal state invalid")

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        # lmdb/h5py objects cannot be pickled; recreate per worker
        state['_env'] = None
        return state

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self._data is not None:
            return self._data[idx]
        if self._meta is not None:
            if self._env is None:
                self._env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
            with self._env.begin() as txn:
                buf = txn.get(f"controls:{idx:08d}".encode())
                if buf is None:
                    raise IndexError(idx)
                x = torch.frombuffer(buf, dtype=torch.float32).view(self.seq_len, self.num_controls)
            return x

        raise RuntimeError("PriorSequenceDataset internal state invalid")


class PriorTokenSequenceDataset(Dataset):
    """Loads discrete token sequences for PriorDiscrete training from LMDB."""

    def __init__(
        self,
        path: Optional[str] = None,
        in_memory: bool = True,
        dtype: torch.dtype = torch.int16,
    ):
        if path is None:
            raise ValueError("PriorTokenSequenceDataset requires 'path'")

        self.path = str(path)
        self.in_memory = bool(in_memory)
        self._dtype = dtype

        self._data: Optional[torch.Tensor] = None
        self._env = None
        self._meta: Optional[dict] = None

        self._init_lmdb()

    def _init_lmdb(self) -> None:
        env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
        with env.begin() as txn:
            meta_buf = txn.get(b"metadata")
            if meta_buf is None:
                raise RuntimeError(f"LMDB token cache missing metadata: {self.path}")
            meta = pickle.loads(meta_buf)

        self._meta = meta
        self.num_sequences = int(meta["num_sequences"])
        self.seq_len = int(meta["seq_len"])
        self._num_codebooks = int(meta["num_codebooks"])
        self._codebook_size = int(meta["codebook_size"])
        self._num_territories = int(meta.get("num_territories", 0) or 0)
        self._territories = None  # [num_sequences] long, when territories are present
        self._cond_envelope = bool(meta.get("cond_envelope", False))
        self._cond_dim = int(meta.get("cond_dim", 0) or 0)
        self._conditions = None  # [num_sequences, seq_len, cond_dim] float, when cond_envelope

        if self.in_memory:
            data = torch.empty((self.num_sequences, self.seq_len, self._num_codebooks), dtype=torch.int16)
            terr = torch.zeros(self.num_sequences, dtype=torch.long) if self._num_territories > 0 else None
            cond = (torch.zeros(self.num_sequences, self.seq_len, self._cond_dim, dtype=torch.float32)
                    if self._cond_envelope and self._cond_dim > 0 else None)
            with env.begin() as txn:
                for i in range(self.num_sequences):
                    buf = txn.get(f"tokens:{i:08d}".encode())
                    if buf is None:
                        raise RuntimeError(f"Missing key tokens:{i:08d} in {self.path}")
                    arr = torch.frombuffer(buf, dtype=self._dtype)
                    data[i] = arr.view(self.seq_len, self._num_codebooks).to(torch.int16)
                    if terr is not None:
                        tb = txn.get(f"terr:{i:08d}".encode())
                        if tb is not None:
                            terr[i] = int(torch.frombuffer(tb, dtype=torch.int16)[0])
                    if cond is not None:
                        cb = txn.get(f"cond:{i:08d}".encode())
                        if cb is not None:
                            cond[i] = torch.frombuffer(bytearray(cb), dtype=torch.float32).view(self.seq_len, self._cond_dim)
            self._data = data
            self._territories = terr
            self._conditions = cond
            try:
                self._data.share_memory_()
                if self._territories is not None:
                    self._territories.share_memory_()
                if self._conditions is not None:
                    self._conditions.share_memory_()
            except RuntimeError:
                pass
            env.close()
        else:
            self._env = env

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            finally:
                self._env = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_env'] = None
        return state

    @property
    def num_codebooks(self) -> int:
        return int(self._num_codebooks)

    @property
    def codebook_size(self) -> int:
        return int(self._codebook_size)

    @property
    def num_territories(self) -> int:
        return int(self._num_territories)

    @property
    def cond_dim(self) -> int:
        return int(self._cond_dim) if self._cond_envelope else 0

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self._data is not None:
            tok = self._data[idx].to(torch.long)
            c = self._conditions is not None
            t = self._territories is not None
            if c and t:
                return tok, self._conditions[idx], int(self._territories[idx])
            if c:
                return tok, self._conditions[idx]
            if t:
                return tok, int(self._territories[idx])
            return tok
        if self._meta is not None:
            if self._env is None:
                self._env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
            with self._env.begin() as txn:
                buf = txn.get(f"tokens:{idx:08d}".encode())
                if buf is None:
                    raise IndexError(idx)
                x = torch.frombuffer(buf, dtype=self._dtype).view(self.seq_len, self._num_codebooks).to(torch.long)
                cond = None
                if self._cond_envelope and self._cond_dim > 0:
                    cb = txn.get(f"cond:{idx:08d}".encode())
                    if cb is not None:
                        cond = torch.frombuffer(bytearray(cb), dtype=torch.float32).view(self.seq_len, self._cond_dim)
                terr = None
                if self._num_territories > 0:
                    tb = txn.get(f"terr:{idx:08d}".encode())
                    terr = int(torch.frombuffer(tb, dtype=torch.int16)[0]) if tb is not None else 0
            if cond is not None and terr is not None:
                return x, cond, terr
            if cond is not None:
                return x, cond
            if terr is not None:
                return x, terr
            return x

        raise RuntimeError("PriorTokenSequenceDataset internal state invalid")
