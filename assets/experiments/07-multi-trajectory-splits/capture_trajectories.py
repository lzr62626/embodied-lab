"""Opt-in collection of seeds43/44 only; offline reuse uses inspect_splits.py."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from capture_protocol import save_ledger


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect",action="store_true",help="Explicitly start this bounded capture job, never used for offline reproduction.")
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--log-dir",type=Path,required=True)
    parser.add_argument("--reference-env",type=Path,required=True)
    parser.add_argument("--cache-dir",type=Path,required=True)
    args=parser.parse_args()
    if not args.collect:parser.error("Collection is opt-in. Use inspect_splits.py for offline reproduction.")
    args.log_dir.mkdir(parents=True,exist_ok=True)
    ledger_path=args.log_dir/"capture-budget.json"
    if ledger_path.exists() or any((args.output_root/f"seed-{seed:04d}").exists() for seed in (43,44)):
        raise FileExistsError("Do not repeat a started collection or overwrite a seed directory.")
    lock=args.log_dir/"collection-started.lock"
    with lock.open("x",encoding="utf-8") as stream:stream.write("One bounded collection; preserve after completion or failure.\n")
    ledger={"format":"bounded_capture_attempts_v1","started_at_utc":datetime.now(timezone.utc).isoformat(),
            "status":"started","max_attempts":40,"attempted_steps_total":0,"active_seed":None,
            "seeds":{str(seed):{"status":"planned","attempted_steps":0,"completed_steps":0} for seed in (43,44)},"attempts":[],
            "note":"Reservations are persisted before env.step and counted conservatively if interrupted. Seed42 is never launched."}
    save_ledger(ledger_path,ledger)
    child_env=dict(os.environ,PYTHONUTF8="1",PYTHONDONTWRITEBYTECODE="1",SIM_CAMERA_CACHE=str(args.cache_dir.resolve()),MS_SKIP_ASSET_DOWNLOAD_PROMPT="1")
    for key,folder in (("MS_ASSET_DIR","maniskill-assets"),("HF_HOME","huggingface"),("TORCH_HOME","torch"),("MPLCONFIGDIR","matplotlib")):
        child_env[key]=str((args.cache_dir/folder).resolve())
    worker=Path(__file__).resolve().parent/"capture_one_trajectory.py"
    for seed in (43,44):
        ledger=json.loads(ledger_path.read_text(encoding="utf-8"));ledger["active_seed"]=seed;ledger["seeds"][str(seed)]["status"]="running";save_ledger(ledger_path,ledger)
        directory=args.output_root/f"seed-{seed:04d}"
        command=[sys.executable,"-B",str(worker),"--seed",str(seed),"--steps","20","--output-dir",str(directory),
                 "--reference-env",str(args.reference_env),"--budget-ledger",str(ledger_path)]
        print(json.dumps({"event":"start_seed","seed":seed,"max_step_attempts":20}),flush=True)
        with (args.log_dir/f"seed-{seed:04d}-r001.log").open("x",encoding="utf-8") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=child_env)
        ledger=json.loads(ledger_path.read_text(encoding="utf-8"));item=ledger["seeds"][str(seed)];item["returncode"]=result.returncode
        env_path=directory/"environment.json"
        if env_path.exists():
            report=json.loads(env_path.read_text(encoding="utf-8"));item["completed_steps"]=report["experiment"]["completed_steps"]
            item["capture_status"]=report["status"]
        item["status"]="complete" if result.returncode==0 else "failed"
        ledger["active_seed"]=None
        if result.returncode:
            ledger["status"]="failed_no_retry";save_ledger(ledger_path,ledger)
            print(json.dumps({"event":"capture_failed","seed":seed,"attempted_total":ledger["attempted_steps_total"]}),flush=True)
            return 1
        save_ledger(ledger_path,ledger)
        print(json.dumps({"event":"complete_seed","seed":seed,"attempts":item["attempted_steps"],"completed":item["completed_steps"]}),flush=True)
    ledger=json.loads(ledger_path.read_text(encoding="utf-8"));ledger["status"]="complete";save_ledger(ledger_path,ledger)
    print(json.dumps({"result":"CAPTURED","new_seed_count":2,"attempted_total":ledger["attempted_steps_total"],"seed42_reused":True}),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
