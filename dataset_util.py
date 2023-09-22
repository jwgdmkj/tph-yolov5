import os

# Base dataset directory
dataset_dir = "/data/dataset/drone_veiw_dataset/SODA_D"

# Subdirectories
subdirs = ["train", "test", "val"]

for subdir in subdirs:
    # Path to the old labels and new labels directories
    labels_old_dir = os.path.join(dataset_dir, subdir, "labels_old")
    labels_new_dir = os.path.join(dataset_dir, subdir, "labels")

    # Create the new labels directory if it doesn't exist
    if not os.path.exists(labels_new_dir):
        os.makedirs(labels_new_dir)

    # Iterate over each txt file in the old labels directory
    for txt_file in os.listdir(labels_old_dir):
        old_file_path = os.path.join(labels_old_dir, txt_file)
        new_file_path = os.path.join(labels_new_dir, txt_file)

        # Read the content and remove duplicates
        with open(old_file_path, "r") as f:
            lines = f.readlines()
            unique_lines = list(set(lines))

        # Write the cleaned content to the new txt file
        with open(new_file_path, "w") as f:
            f.writelines(unique_lines)

print("Duplicate ground truths removed.")