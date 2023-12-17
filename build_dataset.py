import cv2
import random
import numpy as np
from perturbations import (
    rnd_perturbations,
    get_rndnumber_as_picture,
    get_negative_rndnumbers_as_picture,
    get_largeint_as_picture,
    set_background_and_rotations,
    visualize_bboxes,
)
import PIL.Image
from copy import deepcopy
import json
import os

random.seed(42)

total_ds_pics_train = 8000
total_ds_pics_val = 2000
canvas_size = 512
img_size_x, img_size_y = 18 * 2, 25 * 2
input_folder = "./images/"
output_folder = "./datasets/numbers/"
visualize = False   #Visualize created images, bounding Boxes etc

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_folder + "train/"):
    os.makedirs(output_folder + "train/")

if not os.path.exists(output_folder + "val/"):
    os.makedirs(output_folder + "val/")

if not os.path.exists(output_folder + "annotations/"):
    os.makedirs(output_folder + "annotations/")

# Read in Pictures of Numbers 1-9, Dot, Comma and Minus
numbers = []

for i in range(10):
    numbers.append(cv2.imread(input_folder + str(i) + ".png", cv2.IMREAD_GRAYSCALE))

numbers.append(cv2.imread(input_folder + "dot.png", cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread(input_folder + "comma.png", cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread(input_folder + "minus.png", cv2.IMREAD_GRAYSCALE))
numbers = [cv2.resize(i, (img_size_x, img_size_y)) for i in numbers]


def create_dataset(
    total_ds_pics,
    output_folder,
    numbers,
    canvas_size,
    img_size_x,
    img_size_y,
    visualize=False,
    shift_imgoutput_numb=0,
):
    # Create Dataset of decimal numbers consisting of individual digits and decimal point
    id = 0
    annotations = []
    numb_distribution = {"rnd": 0, "neg": 0, "large": 0}

    pic_numb = 0

    while pic_numb < total_ds_pics:
        # Generate random float number between 10000 and 0 with random precision

        cutoff = random.random()

        if cutoff <= 0.2:
            rnd_numb_form, rnd_numb_pic = get_negative_rndnumbers_as_picture(numbers)
            numb_type = "neg"
        elif cutoff <= 0.4:
            rnd_numb_form, rnd_numb_pic = get_largeint_as_picture(numbers)
            numb_type = "large"
        else:
            rnd_numb_form, rnd_numb_pic = get_rndnumber_as_picture(numbers)
            numb_type = "rnd"

        # Alter Pictures with Perturbations
        rnd_numb_pic = rnd_perturbations(rnd_numb_pic)

        # Set Black Background, rotate and save bounding-boxes
        bboxes = []
        for i, pic in enumerate(rnd_numb_pic):
            rnd_numb_pic[i], bbox = set_background_and_rotations(
                pic, img_size_x, img_size_y, multipl=25, resize_imgs=False
            )
            bboxes.append(bbox)

        # Paste all numbers and delimiter onto a 512x512 canvas
        anchor = (
            random.randint(1, max(canvas_size / 4 - img_size_x, 0)),
            random.randint(1, canvas_size - img_size_y),
        )

        bg_black_all = PIL.Image.new("RGB", (canvas_size, canvas_size))

        for i, pic in enumerate(rnd_numb_pic):
            image = PIL.Image.fromarray(np.uint8(pic)).convert("RGB")
            bg_black_all.paste(image, anchor)
            # print("Picture and current anchor: ", pic.shape, anchor)
            bboxes[i][0] += anchor[0]
            bboxes[i][1] += anchor[1]
            anchor = (anchor[0] + pic.shape[1], anchor[1])

        # Visualize Bounding Boxes
        if visualize:
            visualize_bboxes(bg_black_all, bboxes)

        skip_image = False
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])

            if end_point[0] > canvas_size or end_point[1] > canvas_size:
                print("Bounding Box is too big!")
                skip_image = True
                break
            # print(start_point, end_point)

        if skip_image:
            print("Skipping Image: ", pic_numb, " due to BBox Size!")
            continue

        # Count for Statistics
        numb_distribution[numb_type] += 1

        rnd_numb_pic_enum = []

        for i in rnd_numb_form:
            if i == ".":
                rnd_numb_pic_enum.append(11)
            elif i == ",":
                rnd_numb_pic_enum.append(12)
            elif i == "-":
                rnd_numb_pic_enum.append(13)
            else:
                rnd_numb_pic_enum.append(int(i) + 1)

        # print(rnd_numb_pic_enum)

        for i, bbox in enumerate(bboxes):
            annotations.append(
                {
                    "area": int(bbox[3] * bbox[2]),
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "category_id": rnd_numb_pic_enum[i],
                    "image_id": pic_numb,
                    "iscrowd": 0,
                    "id": id,
                    "segmentation": [],
                }
            )
            id += 1

        # for i in annotations:
        #     print(i)
        print("Saving Image: ", pic_numb)
        image = cv2.bitwise_not(np.asarray(bg_black_all))
        cv2.imwrite(
            output_folder + str(pic_numb + shift_imgoutput_numb) + ".jpg", image
        )
        pic_numb += 1

    return annotations, numb_distribution


annotations_train, numbers_distribution_train = create_dataset(
    total_ds_pics_train,
    output_folder + "train/",
    numbers,
    canvas_size,
    img_size_x,
    img_size_y,
    visualize,
)
annotations_val, numbers_distribution_val = create_dataset(
    total_ds_pics_val,
    output_folder + "val/",
    numbers,
    canvas_size,
    img_size_x,
    img_size_y,
    visualize,
    total_ds_pics_train,
)


def create_annotations_json(
    annotations, total_ds_pics, output_name, shift_imgoutput_numb=0
):
    # Additional Information according to COCO Style - various rnd / dummy choices

    categories = []

    for i in range(1, 11):
        categories.append({"id": i, "name": str(i - 1), "supercategory": None})

    categories.append({"id": 11, "name": ".", "supercategory": None})

    images = []

    for i in range(total_ds_pics):
        images.append(
            {
                "coco_url": "",
                "date_captured": "2023-11-11 01:45:07.508146",
                "file_name": str(i + shift_imgoutput_numb) + ".jpg",
                "flickr_url": "",
                "height": 512,
                "id": i,
                "license": 1,
                "width": 512,
            }
        )

    info = {
        "contributor": "",
        "data_created": "2023-11-11",
        "description": "",
        "url": "",
        "version": "",
        "year": 2023,
    }

    licenses = [{"id": 1, "name": None, "url": None}]

    numbers_dataset = {
        "annotations": annotations,
        "categories": categories,
        "images": images,
        "info": info,
        "licenses": licenses,
    }

    # Save to json file
    with open(output_name, "w") as f:
        json.dump(numbers_dataset, f)


create_annotations_json(
    annotations_train,
    total_ds_pics_train,
    output_folder + "annotations/instances_train.json",
)
create_annotations_json(
    annotations_val,
    total_ds_pics_val,
    output_folder + "annotations/instances_val.json",
    total_ds_pics_train,
)

print("Numbers Distribution Training-Set: ", numbers_distribution_train)
print("Numbers Distribution Validation-Set: ", numbers_distribution_val)
