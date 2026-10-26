"""Portable numerical provenance and strict trajectory-group split checks."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

MODEL_KEYS=("obs/agent/qpos","obs/agent/qvel","obs/sensor_data/base_camera/rgb","actions")


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def numeric_fingerprint(arrays,keys=None):
    """v1: sorted keys, shape, little-endian dtype and C-order value bytes.

    ZIP timestamps/order/compression and filesystem paths are not hashed.
    Model-content fingerprints deliberately exclude labels/diagnostics so a
    renamed/repacked input duplicate cannot hide behind changed bookkeeping.
    """
    h=sha256(b"trajectory-numeric-content-v1\0")
    for key in sorted(arrays if keys is None else keys):
        value=np.asarray(arrays[key])
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("Only finite real numeric arrays are accepted: "+key)
        canonical=value.astype(value.dtype.newbyteorder("<"),copy=False)
        header=json.dumps({"key":key,"shape":list(value.shape),"dtype":canonical.dtype.str},sort_keys=True,separators=(",",":")).encode()
        h.update(len(header).to_bytes(8,"little"));h.update(header);h.update(canonical.tobytes(order="C"))
    return h.hexdigest()


def read_numeric(path):
    with np.load(path,allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def validate_split_assignment(manifest,splits):
    records=manifest["trajectories"]
    ids=[r["trajectory_id"] for r in records]
    sources=[r["source_id"] for r in records]
    if len(set(ids))!=len(ids) or len(set(sources))!=len(sources):
        raise ValueError("Trajectory IDs and source IDs must be unique.")
    if set(splits)!= {"train","validation"} or not splits["train"] or not splits["validation"]:
        raise ValueError("Exactly two nonempty splits are required: train and validation.")
    flat=[key for name in ("train","validation") for key in splits[name]]
    if len(flat)!=len(set(flat)) or set(flat)!=set(ids):
        raise ValueError("Every trajectory must belong to exactly one split; no missing/overlapping sources.")
    content=[r["model_content_fingerprint"] for r in records]
    if len(set(content))!=len(content):
        raise ValueError("Duplicate numerical model content; relabeling or repacking does not create a new source.")


def source_path(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("Source path must remain inside the trajectory directory.")
    return path
