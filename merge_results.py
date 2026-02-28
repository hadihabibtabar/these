import os

import pandas as pd

submission_name = "final_submission"
submission_dir = "./submission"
os.makedirs(submission_dir, exist_ok=True)

ensemble_files = [
    "submission_MMoE_profile_a.csv",
    "submission_MMoE_profile_b.csv",
    "submission_MMoE_profile_c.csv",
    "submission_MMoE_profile_c_balanced.csv",
    "submission_MMoE_profile_soft.csv",
]

frames = []
for file_name in ensemble_files:
    path = os.path.join(submission_dir, file_name)
    if os.path.exists(path):
        frames.append(pd.read_csv(path, sep="\t").sort_values("RowId").reset_index(drop=True))

if not frames:
    raise FileNotFoundError("No submission files were found for ensembling.")

final = sum(frames) / len(frames)
final["RowId"] = final["RowId"].astype(int)
final.to_csv(os.path.join(submission_dir, f"{submission_name}.csv"), sep="\t", index=False)
