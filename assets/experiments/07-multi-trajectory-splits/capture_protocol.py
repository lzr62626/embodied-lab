"""Single-writer, durable attempt reservations for the bounded capture job."""
import json
import os
from pathlib import Path


def save_ledger(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def reserve_step(path,seed,step):
    path=Path(path);ledger=json.loads(path.read_text(encoding="utf-8"))
    key=str(seed);item=ledger["seeds"][key]
    if seed not in (43,44) or ledger["active_seed"]!=seed or item["status"]!="running":
        raise RuntimeError("This seed has no active capture authorization.")
    if step!=item["attempted_steps"]+1 or step>20 or ledger["attempted_steps_total"]>=40:
        raise RuntimeError("Step reservation would exceed the fixed capture budget.")
    item["attempted_steps"]=step
    ledger["attempted_steps_total"]+=1
    ledger["attempts"].append({"seed":seed,"step":step})
    save_ledger(path,ledger)  # before env.step, conservatively consumed even if interrupted
