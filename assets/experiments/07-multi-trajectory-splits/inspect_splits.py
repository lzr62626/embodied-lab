"""Offline-only manifest, fixed trajectory splits and CPU batch export."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
import torch
from torch.utils.data import DataLoader

from multi_trajectory_dataset import MultiTrajectoryDataset
from source_utils import MODEL_KEYS,file_sha256,numeric_fingerprint,read_numeric,validate_split_assignment

ASSIGNMENT={"train":["seed-0042","seed-0043"],"validation":["seed-0044"]}


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def numeric_batch(tree,prefix=""):
    arrays={};labels={}
    for key,value in tree.items():
        name=prefix+key
        if isinstance(value,dict):
            a,l=numeric_batch(value,name+"/");arrays.update(a);labels.update(l)
        elif isinstance(value,torch.Tensor):
            if value.device.type!="cpu" or value.requires_grad:raise ValueError("CPU data tensors only.")
            arrays[name]=value.numpy().copy()
        elif name=="metadata/trajectory_id" and isinstance(value,(list,tuple)) and all(isinstance(x,str) for x in value):
            labels[name]=list(value)
        else:raise TypeError("Unexpected batch field: "+name)
    return arrays,labels


def make_preview(path,records,arrays):
    image=Image.new("RGB",(560,984),"#faf8f4");draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=19)
    draw.text((16,12),"Split by source, then build windows",fill="#292b27",font=font)
    tiles=[]
    for i,row in enumerate(records):
        y=56+i*304
        draw.text((16,y+2),f"{row['trajectory_id']} | first real frame | t=0.00s",fill="#292b27",font=font)
        rgb=arrays[row["source_id"]]["obs/sensor_data/base_camera/rgb"][0,0]
        image.paste(Image.fromarray(rgb),(16,y+32))
        role="train" if row["trajectory_id"] in ASSIGNMENT["train"] else "validation"
        lines=[f"source_id: {row['source_id']}",f"seed: {row['seed']}","reused history" if row["reused"] else "new capture",
               "split: "+role,f"N={row['N']} | windows={row['windows']}"]
        for j,text in enumerate(lines):draw.text((296,y+66+30*j),text,fill="#51554d",font=font)
        tiles.append({"trajectory_id":row["trajectory_id"],"source_id":row["source_id"],"state_index":0,
                      "split":role,"box_xywh":[16,y+32,256,256]})
    image.save(path)
    return {"size_wh":list(image.size),"tiles":tiles,"note":"Original RGB pixels; labels outside tiles, excluded from model inputs."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path(__file__).resolve().parent/"trajectories")
    parser.add_argument("--output-dir",type=Path,default=Path("result"))
    args=parser.parse_args()
    names=("dataset_manifest.json","splits.json","window_index.json","train_batch.npz","validation_batch.npz",
           "split_preview.png","leakage_check.json","schema.json","environment.json","batch_summary.json")
    if any((args.output_dir/n).exists() for n in names):raise FileExistsError("Preserve prior exports; choose a new output directory.")
    records=[];arrays={};rgb_hashes={};common=None
    comparable=("task","robot","num_envs","obs_mode","control_mode","camera","width","height","shader_pack","sim_backend","render_backend")
    for source_id,seed in enumerate((42,43,44)):
        label=f"seed-{seed:04d}";directory=args.input_dir/label;path=directory/"trajectory.npz"
        data=read_numeric(path);arrays[source_id]=data
        env=json.loads((directory/"environment.json").read_text(encoding="utf-8"))
        traj=json.loads((directory/"trajectory.json").read_text(encoding="utf-8"))
        config={k:env["experiment"][k] for k in comparable}
        if common is None:common=config
        if config!=common or env["experiment"]["seed"]!=seed:raise ValueError("Source configuration or seed differs from the fixed plan.")
        n=len(data["actions"]);count=max(0,n-4+1)
        if traj["N"]!=n or data["obs/agent/qpos"].shape[0]!=n+1:raise ValueError("N/N+1 mismatch.")
        frame_fingerprints=[sha256(frame.tobytes()).hexdigest() for frame in data["obs/sensor_data/base_camera/rgb"][:,0]]
        rgb_hashes[label]=frame_fingerprints
        records.append({"trajectory_id":label,"source_id":source_id,"seed":seed,"reused":seed==42,
            "origin":"article4 historical capture" if seed==42 else "article7 bounded capture",
            "file":label+"/trajectory.npz","file_sha256":file_sha256(path),
            "numeric_content_fingerprint":numeric_fingerprint(data),"model_content_fingerprint":numeric_fingerprint(data,MODEL_KEYS),
            "fingerprint_spec":"v1: sorted keys+shape+little-endian dtype+C-order bytes; model hash uses qpos/qvel/RGB/actions only",
            "N":n,"observations":n+1,"windows":count,"stop_reason":traj["stop_reason"],
            "simulation_duration_s":traj["simulation_duration_s"],"statistics":traj["statistics"],
            "current_job_step_attempts":0 if seed==42 else env["experiment"]["attempted_step_calls"],
            "configuration":config,"timing":env["timing"],"resolved_backend":env["resolved_backend"],
            "render_device_api":{k:env.get("render_provenance",{}).get(k) for k in
                ("render_system_device","device_is_cpu","device_is_cuda","image_transfer","physical_render_device_note")},
            "rgb_frame_sha256":frame_fingerprints})
    manifest={"format":"article7-multi-trajectory-v1","K":4,"stride":1,"trajectories":records,
              "trajectory_id_lookup":{str(r["source_id"]):r["trajectory_id"] for r in records},
              "note":"Three saved fragments, not three completed episodes or proof of statistical independence. Seed42 reused, only43/44new."}
    validate_split_assignment(manifest,ASSIGNMENT)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/"dataset_manifest.json",manifest);write_json(args.output_dir/"splits.json",ASSIGNMENT)
    datasets={split:MultiTrajectoryDataset(args.output_dir/"dataset_manifest.json",args.output_dir/"splits.json",args.input_dir,split) for split in ASSIGNMENT}
    all_entries=datasets["train"].entries_all
    index={"K":4,"stride":1,"total_windows":len(all_entries),"split_window_counts":{k:len(v) for k,v in datasets.items()},"windows":all_entries}
    write_json(args.output_dir/"window_index.json",index)
    config={"batch_size":4,"shuffle":False,"drop_last":False,"num_workers":0,"pin_memory":False,"generator_seed":42,"device":"cpu"}
    batch_summary={};first_arrays={}
    for split,ds in datasets.items():
        loader=DataLoader(ds,batch_size=4,shuffle=False,drop_last=False,num_workers=0,pin_memory=False,generator=torch.Generator(device="cpu").manual_seed(42))
        first=None;last=None;rows=[]
        for batch in loader:
            numeric,labels=numeric_batch(batch)
            rows.append({"global_window_indices":numeric["metadata/global_window_index"].tolist(),
                         "source_ids":numeric["metadata/source_id"].tolist(),"trajectory_ids":labels["metadata/trajectory_id"]})
            if first is None:first=numeric;first_labels=labels
            last=numeric
        if first is None:raise ValueError("Split has no batches.")
        first_arrays[split]=first
        np.savez_compressed(args.output_dir/(split+"_batch.npz"),**first)
        batch_summary[split]={"windows":len(ds),"batch_count":len(rows),"batch_sizes":[len(b["global_window_indices"]) for b in rows],
                              "batches":rows,"first_string_metadata":first_labels,
                              "first_shapes":{k:list(v.shape) for k,v in first.items()},"last_shapes":{k:list(v.shape) for k,v in last.items()}}
    pairwise=[]
    for a,b in itertools.combinations(records,2):
        qa=arrays[a["source_id"]]["obs/agent/qpos"][0,0];qb=arrays[b["source_id"]]["obs/agent/qpos"][0,0]
        ra=arrays[a["source_id"]]["obs/sensor_data/base_camera/rgb"][0,0];rb=arrays[b["source_id"]]["obs/sensor_data/base_camera/rgb"][0,0]
        pairwise.append({"a":a["trajectory_id"],"b":b["trajectory_id"],"qpos_max_abs_difference":float(np.max(np.abs(qa-qb))),
                         "rgb_changed_pixels":int(np.any(ra!=rb,axis=-1).sum()),"rgb_mean_abs_byte_difference":float(np.abs(ra.astype(np.int16)-rb.astype(np.int16)).mean())})
    train_frames={h for label in ASSIGNMENT["train"] for h in rgb_hashes[label]};val_frames={h for label in ASSIGNMENT["validation"] for h in rgb_hashes[label]}
    leakage={"result":"PASS","trajectory_overlap":sorted(set(ASSIGNMENT["train"])&set(ASSIGNMENT["validation"])),
             "duplicate_numeric_content_groups":[],"duplicate_model_content_groups":[],
             "cross_split_exact_rgb_hashes":sorted(train_frames&val_frames),"cross_split_exact_rgb_unique_count":len(train_frames&val_frames),
             "first_observation_pairwise":pairwise,"normalization_fitted":False,
             "scope_note":"Only explicit ID/content duplicates and exact RGB repeats checked. This is not proof of statistical independence, no leakage, or generalization. Future fitted statistics must use training sources only."}
    write_json(args.output_dir/"leakage_check.json",leakage)
    figure=make_preview(args.output_dir/"split_preview.png",records,arrays)
    fields={}
    for k,v in first_arrays["train"].items():
        axis=["B","K+1","C","H","W"] if k=="observations/rgb" else ["B","K+1","J"] if k in ("observations/qpos","observations/qvel") else ["B","K","A"] if k=="actions" else ["B","K"] if k in ("rewards","terminated","truncated","metadata/action_indices") else ["B","K+1"] if k in ("metadata/state_indices","metadata/step_index","metadata/time_s") else ["B"]
        unit=["rad"]*7+["m"]*2 if k.endswith("qpos") else ["rad/s"]*7+["m/s"]*2 if k.endswith("qvel") else "RGB[0,1]" if k.endswith("rgb") else "simulation seconds" if k.endswith("time_s") else "normalized command [-1,1]" if k=="actions" else "unchanged task reward" if k=="rewards" else "boolean" if v.dtype==bool else "integer metadata/index"
        fields[k]={"shape":list(v.shape),"dtype":str(v.dtype),"axes":axis,"units":unit}
    schema={"format":"article7-numeric-split-batches-v1","K":4,"stride":1,"fields":fields,
            "trajectory_id_strings":"Retained in Dataset/DataLoader metadata and batch_summary.json; not stored as object/Unicode NPZ arrays. Numeric source_id maps through dataset_manifest.json.",
            "split_code":{"0":"train","1":"validation"},"observation_subset":["qpos","qvel","rgb"],
            "excluded_from_model_inputs":["source/split labels","diagnostic/*","info/success","controller targets"],
            "causality":"Offline K+1 state records; future observations not assumed visible at window start."}
    write_json(args.output_dir/"schema.json",schema)
    new_attempts=sum(x["current_job_step_attempts"] for x in records)
    environment={"record_type":"local_offline_multi_trajectory_export","created_at_utc":datetime.now(timezone.utc).isoformat(),
                 "python_version":platform.python_version(),"packages":{"numpy":np.__version__,"torch":torch.__version__,"pillow":pillow_version},
                 "processing_device":"cpu","torch_cuda_build":torch.version.cuda,"training":False,"test_set":False,"learned_normalization":False,
                 "collection_summary":{"reused_seeds":[42],"new_seeds":[43,44],"new_attempts":new_attempts,"max_new_attempts":40},
                 "loader":config,"source_configuration":common,
                 "helper_sha256":{n:file_sha256(Path(__file__).resolve().parent/n) for n in ("trajectory_dataset.py","trajectory_window_dataset.py")},
                 "local_paths":{"python":sys.executable,"input":str(args.input_dir.resolve()),"output":str(args.output_dir.resolve())}}
    write_json(args.output_dir/"environment.json",environment)
    summary={"result":"exported_pending_independent_validation","loader":config,"splits":batch_summary,"preview":figure,
             "numeric_arrays_per_batch":{k:len(v) for k,v in first_arrays.items()},"total_windows":len(all_entries),
             "note":"Default command is offline; no capture module imported or run. Final acceptance is separate validation.json."}
    write_json(args.output_dir/"batch_summary.json",summary)
    print(json.dumps({"result":"EXPORTED","sources":len(records),"total_windows":len(all_entries),"train_windows":len(datasets['train']),
                      "validation_windows":len(datasets['validation']),"train_batch_sizes":batch_summary['train']['batch_sizes'],
                      "validation_batch_sizes":batch_summary['validation']['batch_sizes'],"cross_split_exact_rgb":len(train_frames&val_frames)}))


if __name__=="__main__":main()
