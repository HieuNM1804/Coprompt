import os
import re
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate CoPrompt results")
    parser.add_argument("--exp-name", type=str, default="CoPrompt_Result", help="Experiment name")
    parser.add_argument("--dataset", type=str, default="caltech101", help="Dataset name")
    parser.add_argument("--shots", type=int, default=16, help="Number of shots")
    parser.add_argument("--trainer", type=str, default="CoPrompt", help="Trainer name")
    parser.add_argument("--cfg", type=str, default="coprompt", help="Config name")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="List of seeds to aggregate")
    return parser.parse_args()

import glob

def get_accuracy_from_log(log_dir):
    if not os.path.exists(log_dir):
        return None
    
    # Find all log files (log.txt, log.txt-2026-04-15-XXX) and sort by modification time, newest first
    log_files = glob.glob(os.path.join(log_dir, "log.txt*"))
    if not log_files:
        return None
        
    latest_log_path = max(log_files, key=os.path.getmtime)
    
    with open(latest_log_path, "r") as f:
        content = f.read()
    
    # regex matches: * accuracy: 85.5%
    matches = re.findall(r"\* accuracy: (\d+\.\d+)%", content)
    if matches:
        return float(matches[-1]) # Return the last accuracy entry
    return None

def main():
    args = parse_args()
    seeds = args.seeds
    
    base_accs = []
    novel_accs = []
    hms = []

    print(f"\nAggregating results for Experiment: {args.exp_name}")
    print(f"{'Seed':<10} | {'Base':<10} | {'Novel':<10} | {'HM':<10}")
    print("-" * 50)

    for seed in seeds:
        # Paths for Base and Novel results
        base_dir = os.path.join("output", args.exp_name, "train_base", args.dataset, f"shots_{args.shots}", args.trainer, args.cfg, f"seed{seed}")
        novel_dir = os.path.join("output", args.exp_name, "test_new", args.dataset, f"shots_{args.shots}", args.trainer, args.cfg, f"seed{seed}")

        base_acc = get_accuracy_from_log(base_dir)
        novel_acc = get_accuracy_from_log(novel_dir)

        if base_acc is not None and novel_acc is not None:
            hm = 2 * (base_acc * novel_acc) / (base_acc + novel_acc)
            base_accs.append(base_acc)
            novel_accs.append(novel_acc)
            hms.append(hm)
            print(f"{seed:<10} | {base_acc:<10.2f} | {novel_acc:<10.2f} | {hm:<10.2f}")
        else:
            status = []
            if base_acc is None: status.append("Base log missing")
            if novel_acc is None: status.append("Novel log missing")
            print(f"{seed:<10} | {', '.join(status)}")

    if base_accs:
        print("-" * 50)
        print(f"{'MEAN':<10} | {np.mean(base_accs):<10.2f} | {np.mean(novel_accs):<10.2f} | {np.mean(hms):<10.2f}")
        print(f"{'STD':<10} | {np.std(base_accs):<10.2f} | {np.std(novel_accs):<10.2f} | {np.std(hms):<10.2f}")
    else:
        print("\n[ERROR] No valid results found to aggregate. Please ensure the experiments finished successfully.")

if __name__ == "__main__":
    main()
