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
    get_word_as_picture,
)
import PIL.Image
from copy import deepcopy
import json
import os

random.seed(42)

total_ds_pics_train = 21000
total_ds_pics_val = 4000
canvas_size = 512
img_size_x, img_size_y = 18 * 2, 25 * 2
input_folder = "./bbg_numbers/"
output_folder = "./datasets/letters_large/"
visualize = False

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_folder + "train/"):
    os.makedirs(output_folder + "train/")

if not os.path.exists(output_folder + "val/"):
    os.makedirs(output_folder + "val/")

if not os.path.exists(output_folder + "annotations/"):
    os.makedirs(output_folder + "annotations/")

# Read in Pictures of Numbers 1-9, Dot, Comma and Minus
letters = []

for i in range(10):
    letters.append(cv2.imread(input_folder + str(i) + ".png", cv2.IMREAD_GRAYSCALE))

letters.append(cv2.imread(input_folder + "dot.png", cv2.IMREAD_GRAYSCALE))
letters.append(cv2.imread(input_folder + "comma.png", cv2.IMREAD_GRAYSCALE))
letters.append(cv2.imread(input_folder + "minus.png", cv2.IMREAD_GRAYSCALE))

count_numbers = len(letters)

for i in range(65, 91):
    letters.append(cv2.imread(input_folder + chr(i) + ".png", cv2.IMREAD_GRAYSCALE))

letters = [cv2.resize(i, (img_size_x, img_size_y)) for i in letters]


def create_dataset(
    total_ds_pics,
    output_folder,
    letters,
    canvas_size,
    img_size_x,
    img_size_y,
    visualize=False,
    shift_imgoutput_numb=0,
):
    # Create Dataset of decimal numbers consisting of individual digits and decimal point
    id = 0
    annotations = []
    numb_distribution = {"rnd": 0, "neg": 0, "large": 0, "word": 0}

    # load dictionary from file dictionary.txt
    with open("dictionary.txt") as f:
        dictionary = f.read().splitlines()

    pic_numb = 0

    while pic_numb < total_ds_pics:
        # Generate random float number between 10000 and 0 with random precision

        get_word_or_num = random.random()

        if get_word_or_num <= 0.7:  # 13Numbers&specials vs. 26 letters
            # if True:
            # Get Word
            word = random.choice(dictionary)
            word = word.upper()
            string_word_numb, pic_word_numb = get_word_as_picture(
                letters[count_numbers:], word
            )
            letter_type = "word"

        else:
            # Get Number
            cutoff = random.random()

            if cutoff <= 0.2:
                string_word_numb, pic_word_numb = get_negative_rndnumbers_as_picture(
                    letters
                )
                letter_type = "neg"
            elif cutoff <= 0.4:
                string_word_numb, pic_word_numb = get_largeint_as_picture(letters)
                letter_type = "large"
            else:
                string_word_numb, pic_word_numb = get_rndnumber_as_picture(letters)
                letter_type = "rnd"

        # Alter Pictures with Perturbations
        pic_word_numb = rnd_perturbations(pic_word_numb)

        # Set Black Background, rotate and save bounding-boxes
        bboxes = []
        for i, pic in enumerate(pic_word_numb):
            pic_word_numb[i], bbox = set_background_and_rotations(
                pic, img_size_x, img_size_y, multipl=25, resize_imgs=True
            )
            bboxes.append(bbox)

        # Paste all numbers and delimiter onto a 512x512 canvas
        anchor = (
            random.randint(1, max(canvas_size / 4 - img_size_x, 0)),
            random.randint(1, canvas_size - img_size_y),
        )

        bg_black_all = PIL.Image.new("RGB", (canvas_size, canvas_size))

        for i, pic in enumerate(pic_word_numb):
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
        numb_distribution[letter_type] += 1

        rnd_numb_pic_enum = []

        if letter_type != "word":
            for i in string_word_numb:
                if i == ".":
                    rnd_numb_pic_enum.append(11)
                elif i == ",":
                    rnd_numb_pic_enum.append(12)
                elif i == "-":
                    rnd_numb_pic_enum.append(13)
                else:
                    rnd_numb_pic_enum.append(int(i) + 1)
        else:
            for i in string_word_numb:
                idx = ord(i) - 65 + count_numbers
                rnd_numb_pic_enum.append(idx + 1)

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
    letters,
    canvas_size,
    img_size_x,
    img_size_y,
    visualize,
)
annotations_val, numbers_distribution_val = create_dataset(
    total_ds_pics_val,
    output_folder + "val/",
    letters,
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
