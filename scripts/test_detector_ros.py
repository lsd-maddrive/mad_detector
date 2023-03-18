#!/usr/bin/env python3

import os

from dcvt.common.utils.fs import get_images_from_directory
from mad_detector import wrapper
from mad_detector import utils


def init_ros():
    import rospy

    rospy.init_node("test_signs_detector")
    # Get args
    args = {"model": rospy.get_param("~model_path"), "input": rospy.get_param("~input")}

    return args


# Resize to minimal 480
TARGET_MIN_SIDE = 480.0


if __name__ == "__main__":
    args = init_ros()
    input_path = args["input"]
    model_path = args["model"]

    RESULT_DIRECTORY = os.path.join(input_path, "predicted")

    try:
        os.makedirs(RESULT_DIRECTORY)
    except:
        pass

    # Execute
    det = wrapper.RFSignsDetector(model_path)
    im_fpaths = get_images_from_directory(input_path)

    for im_fpath in im_fpaths:
        print(f"Processing file: {im_fpath}")

        img = utils.read_image_fs(im_fpath)
        img = utils.resize_to_min_side_size(img, TARGET_MIN_SIDE)

        # Predict (can be any size of input)
        bboxes, labels, scores = det.find_signs(img)

        # Render boxes
        for i_p, bbox in enumerate(bboxes):
            utils.draw_box_with_text(img, bbox=bbox, score=round(scores[i_p], 2), label=labels[i_p])

        result_img_fpath = os.path.join(RESULT_DIRECTORY, os.path.basename(im_fpath))
        utils.write_image_fs(result_img_fpath, img)
