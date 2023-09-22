import os
import json

labels_dir = "/data/dataset/drone_veiw_dataset/SODA_A/labels/val"
annotations_dir = "/data/dataset/drone_veiw_dataset/SODA_A/annotations/val"
img_dir = "/data/dataset/drone_veiw_dataset/SODA_A/images"

def convert_to_yolo_label(data):
    yolo_labels = []

    # Get image dimensions
    img_width = data["images"]["width"]
    img_height = data["images"]["height"]

    for annotation in data["annotations"]:
        category_id = annotation["category_id"]

        # Extract the bounding box from the polygon
        x_coords = annotation["poly"][::2]
        y_coords = annotation["poly"][1::2]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Convert to YOLO format
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize the values
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        yolo_labels.append(f"{category_id} {x_center} {y_center} {width} {height}")

    return yolo_labels

# Process each JSON file in the directory
for json_file in os.listdir(annotations_dir):
    if json_file.endswith(".json"):
        with open(os.path.join(annotations_dir, json_file), 'r') as f:
            data = json.load(f)
            yolo_labels = convert_to_yolo_label(data)

            # Save the YOLO labels to a .txt file with the same name as the JSON file
            output_file = os.path.join(labels_dir, json_file.replace(".json", ".txt"))
            with open(output_file, 'w') as out_f:
                out_f.write("\n".join(yolo_labels))