import os

from dcvt.common.utils.fs import get_images_from_directory
from mad_detector import wrapper
from mad_detector import utils


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test script for signs detector")
    parser.add_argument("-m", action="store", dest="model", required=True)
    parser.add_argument("-i", action="store", dest="input", required=True)
    parser.add_argument("-s", type=int, dest="size", default=480)

    args = parser.parse_args()
    return vars(args)  # As dictionary


if __name__ == "__main__":
    args = get_args()
    input_path = args["input"]
    model_path = args["model"]
    target_min_size = args["size"]

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
        img = utils.resize_to_min_side_size(img, target_min_size)

        # Predict (can be any size of input)
        bboxes, labels, scores = det.find_signs(img)

        # Render boxes
        for i_p, bbox in enumerate(bboxes):
            utils.draw_box_with_text(img, bbox=bbox, score=round(scores[i_p], 2), label=labels[i_p])

        result_img_fpath = os.path.join(RESULT_DIRECTORY, os.path.basename(im_fpath))
        utils.write_image_fs(result_img_fpath, img)
