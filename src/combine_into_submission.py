import os

# edit start

dirs = [
    "2023-10-14_14-00-59_tile-xla_28247423_filter_full",
    "2023-10-14_13-58-37_layout-xla-default_28247343_filter_full",
    "2023-10-13_22-59-32_layout-xla-random_28242995_filter",
    "2023-10-14_13-58-58_layout-nlp-default_28247289_filter_full",
    "2023-10-14_13-58-47_layout-nlp-random_28247324_filter_full",
]

submission_name = "filtered_combined_half_and_full"

# edit end


REPO_ROOT = os.path.realpath(os.path.join(os.path.split(__file__)[0], ".."))

dk_submissions_dir = "dk_submissions"
os.makedirs(dk_submissions_dir, exist_ok=True)
submission_path = f"{dk_submissions_dir}/{submission_name}.csv"

header = "ID,TopConfigs"

with open(submission_path, "w") as fo:
    fo.write(header + "\n")

    for dir in dirs:
        path = f"{REPO_ROOT}/runs/{dir}/submission_test_auto.csv"

        with open(path, "r") as fi:
            lines = fi.readlines()
        
        for line in lines:
            fo.write(line)

print("Combination done")
