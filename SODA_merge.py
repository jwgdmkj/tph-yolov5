import os
import shutil

# Base directories
soda_a_dir = "/data/dataset/drone_veiw_dataset/SODA_A"
soda_d_dir = "/data/dataset/drone_veiw_dataset/SODA_D"
soda_dir = "/data/dataset/drone_veiw_dataset/SODA"

# Subdirectories
subdirs = ["test", "train", "val"]
secondary_dirs = ["images", "labels"]

def rename_and_move_files(src_dir, dest_dir, extension):
    for filename in os.listdir(src_dir):
        if filename.endswith(extension):
            # Extract the base name and the extension
            base_name, ext = os.path.splitext(filename)

            # Add 10000 to the base name
            new_base_name = str(int(base_name) + 10000).zfill(5)

            # Construct the new filename
            new_filename = new_base_name + ext

            # If it's a txt file, modify the class_id
            if ext == ".txt":
                with open(os.path.join(src_dir, filename), 'r') as f:
                    lines = f.readlines()

                with open(os.path.join(dest_dir, new_filename), 'w') as f:
                    for line in lines:
                        parts = line.split()
                        class_id = int(parts[0])

                        # Modify the class_id
                        if class_id == 9:
                            class_id = 9
                        else:
                            class_id += 10

                        # Write the modified line
                        f.write(str(class_id) + " " + " ".join(parts[1:]) + "\n")
            else:
                # Move the file without modification
                shutil.copy(os.path.join(src_dir, filename), os.path.join(dest_dir, new_filename))

# Ensure the SODA directory exists
if not os.path.exists(soda_dir):
    os.makedirs(soda_dir)

# Iterate over each subdirectory and secondary subdirectory
for subdir in subdirs:
    for sec_dir in secondary_dirs:
        src_a_dir = os.path.join(soda_a_dir, subdir, sec_dir)
        src_d_dir = os.path.join(soda_d_dir, subdir, sec_dir)
        dest_dir = os.path.join(soda_dir, subdir, sec_dir)

        # Ensure the destination directory exists
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Copy files from SODA_A to SODA
        # for filename in os.listdir(src_a_dir):
        #     shutil.copy(os.path.join(src_a_dir, filename), os.path.join(dest_dir, filename))

        # Rename and move image and label files from SODA_D to SODA
        # if sec_dir == "images":
        #     rename_and_move_files(src_d_dir, dest_dir, ".jpg")
        if sec_dir == "labels":
            rename_and_move_files(src_d_dir, dest_dir, ".txt")

print("Directories merged successfully.")
