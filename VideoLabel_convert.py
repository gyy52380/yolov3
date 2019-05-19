import os
import shutil
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--directory_path", type=str, help="directory containing images, labels, and classes.txt")
parser.add_argument("--out_directory",  type=str, help="directory where .data, .names, images/, labels/ will be saved")
parser.add_argument("--cfg_template_path", type=str, help="path to yolov3.cfg to be configured for this dataset")
parser.add_argument("--validation_fraction", type=float, default=0.2, help="amount of dataset to be used for validation")
parser.add_argument("--test_fraction", type=float, default=0.05, help="amount of dataset to be used for testing. These images are placed in separate 'test' directory. Run test.py on this folder.")
parser.add_argument("--check_corruption", action="store_true", help="check for corrupted images. Must have opencv installed.")

#opt = parser.parse_args(["--directory_path",    "C:/Users/gabri/Desktop/obj1",
#                         "--out_directory",     "./converted",
#                         "--cfg_template_path", "./cfg/yolov3.cfg",
#                         "--check_corruption"])
opt = parser.parse_args()

if opt.validation_fraction + opt.validation_fraction > 1:
    print("error: validation_fraction + test_fraction > 1.0.")
    exit(0)
if not os.path.isdir(opt.directory_path):
    print("error: '{}' is not a valid path.".format(opt.directory_path))
    exit(0)
if not os.path.exists(opt.cfg_template_path):
    print("error: '{}' is not a valid path.".format(opt.directory_path))
    exit(0)

check_corruption = False
if opt.check_corruption:
    try:
        import cv2
        check_corruption = True
    except:
        print("Error. Can't check for corrupted images because opencv isn't installed.")

print(opt)


directory       = os.path.abspath(opt.directory_path)
out_directory   = os.path.abspath(opt.out_directory)
image_directory = os.path.join(out_directory, "images")
label_directory = os.path.join(out_directory, "labels")
test_directory  = os.path.join(out_directory, "test")

print("Creating directories...")
os.makedirs(opt.out_directory, exist_ok=True)
os.makedirs(image_directory,   exist_ok=True)
os.makedirs(label_directory,   exist_ok=True)
os.makedirs(test_directory,    exist_ok=True)


classes_count = 0

# make .names file
print("Creating dataset.names...")
with open(os.path.join(directory, "classes.txt"), 'r') as classes_file:
    lines = classes_file.readlines()
    classes_count = len(lines)
    with open(os.path.join(out_directory, "dataset.names"), 'w+') as names_file:
        names_file.writelines(lines)

# make a .data file
print("Creating dataset.data...")
with open(os.path.join(out_directory, "dataset.data"), 'w+') as data_file:
    data_file.write("classes={}\n".format(classes_count))
    data_file.write("train={}\n".format(os.path.join(out_directory, "train_list.txt")))
    data_file.write("valid={}\n".format(os.path.join(out_directory, "valid_list.txt")))
    data_file.write("names={}\n".format(os.path.join(out_directory, "dataset.names")))
    data_file.write("backup={}\n".format(os.path.join(out_directory, "backup/")))
    data_file.write("eval=coco\n") # ?

# make yolov3.cfg file
print("Creating dataset_yolov3.cfg...")
with open(opt.cfg_template_path, 'r') as template:
    lines = template.readlines()
    for i in range(len(lines)):
        if lines[i].startswith("classes="):
            lines[i] = "classes={}\n".format(classes_count)

        if lines[i].startswith("[yolo]"):
            for j in range(i, -1, -1):
                if lines[j].startswith("filters="):
                    lines[j] = "filters={}\n".format(15 + 3*classes_count)
                    break

    with open(os.path.join(out_directory, "custom_yolov3.cfg"), 'w+') as cfg_file:
        cfg_file.writelines(lines)


# make list.txt files
print("Copying images and labels...")
train_image_list = []
validation_image_list = []

def is_image(path):
    return  path.endswith(".png") or\
            path.endswith(".bmp") or\
            path.endswith(".jpg") or\
            path.endswith(".jpeg")

for filename in os.listdir(directory):
    abs_filename = os.path.join(directory, filename)
    if is_image(abs_filename):

        if check_corruption:
            #print("checking {}".format(abs_filename))
            #image = cv2.imread(abs_filename)
            if cv2.imread(abs_filename) is None:
                print("Image '{}' is corrupted. Skipping.".format(abs_filename))
                continue

        chance = random.random()

        # add to test directory
        if chance <= opt.test_fraction:
            shutil.copy2(abs_filename, test_directory)

        # add to images directory
        elif chance <= (opt.test_fraction + opt.validation_fraction):
            validation_image_list.append("{}\n".format(os.path.join(test_directory, filename)))
            shutil.copy2(abs_filename, image_directory)
        else:
            train_image_list.append("{}\n".format(os.path.join(image_directory, filename)))
            shutil.copy2(abs_filename, image_directory)

        label_file = abs_filename[:abs_filename.rfind('.')] + ".txt"
        if os.path.exists(label_file):
            shutil.copy2(label_file, label_directory)


with open(os.path.join(out_directory, "train_list.txt"), 'w+') as train_list_file:
    train_list_file.writelines(train_image_list)
with open(os.path.join(out_directory, "valid_list.txt"), 'w+') as valid_list_file:
    valid_list_file.writelines(validation_image_list)

print("Done!")
