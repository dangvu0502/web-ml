import os
import json
import glob


def get_image_data(folder_path):
    # Use glob to find all PNG files in the folder
    image_files = glob.glob(os.path.join(folder_path, "*.png"))

    # Create a list of dictionaries with image path and caption (filename without extension)
    image_data = []
    for image_path in image_files:
        caption = os.path.basename(image_path)  # Get filename
        print(image_path)
        # caption = os.path.splitext(caption)[0]  # Remove the extension
        image_data.append(caption.split("/")[0][:-4])

    return image_data


def save_to_json(data, json_path):
    # Save the data to a JSON file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def main():
    folder_path = "dataset/train"  # Specify your folder path here
    json_path = "train_data.json"  # Specify the output JSON file path

    image_data = get_image_data(folder_path)
    save_to_json(image_data, json_path)
    print(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
