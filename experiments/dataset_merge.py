import os
import csv
from fuzzywuzzy import fuzz  # Install fuzzywuzzy package using pip if not already installed
import argparse
import shutil


def ids_classnames_from_csv(csv_file):
    class_names = []
    class_ids = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            class_id = row[0].strip()[1:-1]  # Assuming class id is in the first column
            class_name = row[1].strip()[1:-1]  # Assuming class name is in the second column
            class_names.append(class_name)
            class_ids.append(class_id)
    return class_ids, class_names


def classnames_from_dir(directory):
    class_names = set()
    for subitem in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subitem)):
            class_names.add(subitem)
    return class_names


def find_similar_classnames(class_names_img, class_names_csv, threshold=80):
    similar_class_names_img = []
    similar_class_names_csv = []
    for name_img in class_names_img:
        for name_csv in class_names_csv:
            similarity_ratio = fuzz.ratio(name_img.lower(), name_csv.lower())
            if similarity_ratio >= threshold:
                similar_class_names_img.append(name_img)
                similar_class_names_csv.append(name_csv)
                break
    return similar_class_names_img, similar_class_names_csv

def find_all_images(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                images.append(os.path.join(root, file))
    return images


def merge(csv_file, image_directory, dest_dir):
    csv_class_ids, csv_class_names = ids_classnames_from_csv(csv_file)
    directory_class_names = classnames_from_dir(image_directory)
    similar_names_img, similar_names_csv = find_similar_classnames(directory_class_names, csv_class_names)

    for directory in directory_class_names:
        if directory in similar_names_img:
            images = find_all_images(os.path.join(image_directory, directory))

            csv_dirname = similar_names_csv[similar_names_img.index(directory)]
            img_class = csv_class_ids[csv_class_names.index(csv_dirname)]

            print(directory, "matches with ", csv_dirname)
            print("Moving ", directory, " contents to ", os.path.join(img_class, img_class))
            for image in images:
                print("Copying ", image, " to ", os.path.join(dest_dir, img_class))
                shutil.copy(image, os.path.join(dest_dir, img_class))
        else:
            print("New class from ", os.path.join(image_directory, directory), " to ", os.path.join(dest_dir, directory))
            try:
                shutil.copytree(os.path.join(image_directory, directory), os.path.join(dest_dir, directory))
            except FileExistsError:
                print("Directory already exists.")


arg = argparse.ArgumentParser(description="Dataset tool for safe merging.")
arg.add_argument("--csv", type=str, help="Path to CSV file with annotations", required=True)
arg.add_argument("--image_dir", type=str, help="Path to directory containing images", required=True)
arg.add_argument("--dest_dir", type=str, help="Path to directory to copy images to", required=True)
args = arg.parse_args()

merge(args.csv, args.image_dir, args.dest_dir)
print("Done merging.")
