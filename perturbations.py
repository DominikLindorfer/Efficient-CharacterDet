import cv2
import numpy as np
import random
import PIL.Image

# Alter Pictures to include perturbations, i.e. a black boarder as well as blurs etc.


# Grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Noise Removal
def remove_noise(image, intensity=1):
    blured = cv2.medianBlur(image, intensity)
    # blured = cv2.resize(blured, (image.shape[1], image.shape[0]))
    return blured


# Thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# Erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# Opening - Erosion followed by Dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def rnd_perturbations(rnd_numb_pic):
    for i, pic in enumerate(rnd_numb_pic):
        if random.random() > 0.2:
            # print("Blured Number: ", i)
            rnd_numb_pic[i] = remove_noise(pic)

        if random.random() > 0.2:
            # print("Tresholded Number: ", i)
            rnd_numb_pic[i] = thresholding(pic)

        # if random.random() > 0.2:
        #     print("Dilated Number: ", i)
        #     rnd_numb_pic[i] = dilate(pic)

    return rnd_numb_pic


def get_rndnumber_as_picture(numbers, max_number=10000, max_prec=3):
    rnd_numb = random.random() * max_number
    prec = random.randint(1, max_prec)

    rnd_numb_formated = "{rnd_numb:.{prec}f}".format(rnd_numb=rnd_numb, prec=prec)
    print(rnd_numb_formated, prec)

    # Patch together a picture of the number above (include perturbations here)
    rnd_numb_pic = []

    for i in rnd_numb_formated:
        if i == ".":
            rnd_numb_pic.append(numbers[10])
        else:
            rnd_numb_pic.append(numbers[int(i)])

    return rnd_numb_formated, rnd_numb_pic


def get_negative_rndnumbers_as_picture(numbers, max_number=1000, max_prec=2):
    rnd_numb = random.random() * max_number * (-1)
    prec = random.randint(1, max_prec)

    rnd_numb_formated = "{rnd_numb:.{prec}f}".format(rnd_numb=rnd_numb, prec=prec)
    print(rnd_numb_formated, prec)

    # Patch together a picture of the number above (include perturbations here)
    rnd_numb_pic = []

    for i in rnd_numb_formated:
        if i == ".":
            rnd_numb_pic.append(numbers[10])
        elif i == "-":
            rnd_numb_pic.append(numbers[12])
        else:
            rnd_numb_pic.append(numbers[int(i)])

    return rnd_numb_formated, rnd_numb_pic


def get_largeint_as_picture(numbers, max_number=1e6):
    rnd_numb = int(random.random() * max_number)

    rnd_numb_formated = "{rnd_numb}".format(rnd_numb=rnd_numb)
    print(rnd_numb_formated)

    # Patch together a picture of the number above (include perturbations here)
    rnd_numb_pic = []

    for i, digit in enumerate(rnd_numb_formated):
        if i == 3:
            rnd_numb_pic.append(numbers[11])
            rnd_numb_formated = rnd_numb_formated[:i] + "," + rnd_numb_formated[i:]

        rnd_numb_pic.append(numbers[int(digit)])

    return list(reversed(rnd_numb_formated)), list(reversed(rnd_numb_pic))


# Set on Background and Rotate? and save bounding-boxes
def set_background_and_rotations(
    image, img_size_x, img_size_y, angle=0, sigma=0.15, multipl=100, resize_imgs=False
):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img_size_x = image.shape[1]
    img_size_y = image.shape[0]

    if resize_imgs:
        factor = min(0.6 + random.random(), 1.4)
        img_size_x = int(img_size_x * factor)
        img_size_y = int(img_size_y * factor)
        image = cv2.resize(image, (img_size_x, img_size_y))

    image = PIL.Image.fromarray(np.uint8(image)).convert("RGB")

    rnd_coords = [
        np.abs(round(random.gauss(0, np.sqrt(sigma)) * multipl)) for i in range(4)
    ]
    # print(rnd_coords)

    # pw, ph = 64, 64
    pw, ph = rnd_coords[0], rnd_coords[1]
    bg_black = PIL.Image.new("RGB", (pw + img_size_x, ph + img_size_y))

    paste_pos = (
        min(rnd_coords[2], pw),
        min(rnd_coords[3], ph),
    )  # must not exceed pw, ph

    bg_black.paste(image, paste_pos)
    bg_black = bg_black.rotate(angle)

    start_point = paste_pos
    end_point = (img_size_x + paste_pos[0], img_size_y + paste_pos[1])

    # DEBUG
    # color = (255, 0, 0)
    # thickness = 2
    # image = cv2.rectangle(np.asarray(bg_black), start_point, end_point, color, thickness)
    # cv2.imshow("test.jpg", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return np.asarray(bg_black), [
        start_point[0],
        start_point[1],
        end_point[0] - start_point[0],
        end_point[1] - start_point[1],
    ]


def visualize_bboxes(image_bg, bboxes):
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])

        color = (255, 0, 0)
        thickness = 2
        image = cv2.rectangle(
            np.asarray(image_bg), start_point, end_point, color, thickness
        )

        cv2.imshow("test.jpg", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Parameters for annotations.json
def create_dataset(
    total_ds_pics,
    output_folder,
    numbers,
    canvas_size,
    IMG_SIZE_X,
    IMG_SIZE_Y,
    visualize=False,
):
    id = 0
    annotations = []

    pic_numb = 0

    while pic_numb < total_ds_pics:
        # Generate random float number between 10000 and 0 with random precision
        rnd_numb_form, rnd_numb_pic = get_rndnumber_as_picture(numbers)

        # Alter Pictures with Perturbations
        rnd_numb_pic = rnd_perturbations(rnd_numb_pic)

        # Set Black Background, rotate and save bounding-boxes
        bboxes = []
        for i, pic in enumerate(rnd_numb_pic):
            rnd_numb_pic[i], bbox = set_background_and_rotations(
                pic, IMG_SIZE_X, IMG_SIZE_Y, multipl=50
            )
            bboxes.append(bbox)

        # Paste all numbers and delimiter onto a 512x512 canvas
        anchor = (
            random.randint(1, max(canvas_size / 4 - IMG_SIZE_X, 0)),
            random.randint(1, canvas_size - IMG_SIZE_Y),
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

        rnd_numb_pic_enum = []

        for i in rnd_numb_form:
            if i == ".":
                rnd_numb_pic_enum.append(11)
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
        cv2.imwrite(output_folder + str(pic_numb) + ".jpg", image)
        pic_numb += 1


def get_word_as_picture(letters, word):
    word_pic = []

    for i in word:
        idx = ord(i) - 65
        word_pic.append(letters[idx])

    return word, word_pic
