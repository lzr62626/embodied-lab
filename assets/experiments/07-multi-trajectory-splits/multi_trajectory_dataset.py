"""Split by trajectory first; concatenate only already-built window indices."""
import json
from operator import index as integer_index
from pathlib import Path

import torch
from torch.utils.data import Dataset

from source_utils import MODEL_KEYS,file_sha256,numeric_fingerprint,read_numeric,source_path,validate_split_assignment
from trajectory_window_dataset import TrajectoryWindowDataset


def checked_window(row,start,stop,source_id,source_length,K):
    """Explicit source/bounds guard, also exercised by private index tests."""
    if source_id!=row["source_id"] or start!=row["start_index"] or stop!=row["end_index_exclusive"]:
        raise ValueError("Window source/bounds mismatch; never cross a trajectory boundary.")
    if not 0<=start<stop<=source_length or stop-start!=K:
        raise ValueError("Window exceeds this source or has an incomplete action horizon.")


class MultiTrajectoryDataset(Dataset):
    def __init__(self,manifest_path,splits_path,trajectory_root,split):
        self.manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.splits=json.loads(Path(splits_path).read_text(encoding="utf-8"))
        validate_split_assignment(self.manifest,self.splits)
        if split not in self.splits:raise ValueError("Unknown split: "+split)
        self.split=split;self.children={};self.entries_all=[]
        self.names={row["source_id"]:row["trajectory_id"] for row in self.manifest["trajectories"]}
        global_index=0
        for row in self.manifest["trajectories"]:
            path=source_path(trajectory_root,row["file"])
            if file_sha256(path)!=row["file_sha256"]:
                raise ValueError("Source file changed: "+row["trajectory_id"])
            arrays=read_numeric(path)
            if numeric_fingerprint(arrays,MODEL_KEYS)!=row["model_content_fingerprint"]:
                raise ValueError("Source numerical content changed: "+row["trajectory_id"])
            child=TrajectoryWindowDataset(path,K=self.manifest["K"],stride=self.manifest["stride"])
            self.children[row["source_id"]]=child
            for local,start in enumerate(child.starts):
                self.entries_all.append({"global_window_index":global_index,"source_id":row["source_id"],
                    "trajectory_id":row["trajectory_id"],"trajectory_window_index":local,
                    "start_index":start,"end_index_exclusive":start+child.K,
                    "split":"train" if row["trajectory_id"] in self.splits["train"] else "validation"})
                global_index+=1
        self.entries=[row for row in self.entries_all if row["trajectory_id"] in self.splits[split]]
        if not self.entries:raise ValueError("This split has no complete windows; do not pad or add captures.")

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        i=integer_index(index)
        if i<0 or i>=len(self):raise IndexError(index)
        row=self.entries[i];child=self.children[row["source_id"]]
        sample=child[row["trajectory_window_index"]]
        m=sample["metadata"]
        checked_window(row,int(m["start_index"]),int(m["end_index_exclusive"]),row["source_id"],child.n,child.K)
        # Article6's single-source ID=0 must be remapped here, without editing it.
        m["source_id"]=torch.tensor(row["source_id"],dtype=torch.int64,device="cpu")
        m["trajectory_window_index"]=m.pop("window_index")
        m["global_window_index"]=torch.tensor(row["global_window_index"],dtype=torch.int64,device="cpu")
        m["split_code"]=torch.tensor(0 if self.split=="train" else 1,dtype=torch.int64,device="cpu")
        m["trajectory_id"]=row["trajectory_id"]  # human ID stays metadata, never model input
        return sample
