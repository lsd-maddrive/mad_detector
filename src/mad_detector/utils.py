import numpy as np
import cv2


def read_image_fs(fpath: str) -> np.array:
    """Read RGB image from filesystem

    Parameters
    ----------
    fpath : str
        Path to read image

    Returns
    -------
    np.array
        RGB image
    """
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_image_fs(fpath: str, img: np.array):
    """Write RGB image to filesystem

    Parameters
    ----------
    fpath : str
        Path to save image in
    img : np.array
        Image to save
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fpath, img)


def resize_to_min_side_size(
    img: np.array, min_size: int, interpolation: int = cv2.INTER_LINEAR
) -> np.array:
    """Resize image with aspect ratio to be minimum size as set

    Parameters
    ----------
    img : np.array
        Image to resize
    min_size : int
        Desired size of minimum size
    interpolation : int, optional
        Interpolation to use, by default cv2.INTER_LINEAR

    Returns
    -------
    np.array
        Resized image
    """
    rsz_ratio = min_size / min(img.shape[:2])
    img = cv2.resize(img, None, fx=rsz_ratio, fy=rsz_ratio, interpolation=interpolation)

    return img


def draw_box_with_text(
    canvas: np.array,
    bbox: list,
    score: float,
    label: str,
    font_sz: float = 0.7,
    font_width: int = 1,
):
    """Draw predictions on image

    Parameters
    ----------
    canvas : np.array
        Image to draw on it
    bbox : list
        Sign bbox
    score : float
        Confidence score
    label : str
        Text to render above
    font_sz : float, optional
        Size of font, by default 0.7
    font_width : int, optional
        Width of font, by default 1
    """
    x, y, w, h = bbox

    cv2.rectangle(
        canvas,
        (int(x), int(y)),
        (int(x + w), int(y + h)),
        color=(0, 0, 255),
        thickness=2,
    )

    cv2.putText(
        canvas,
        text="{} {}".format(label, str(score)),
        org=(int(x), int(y - 5)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_sz,
        color=(0, 0, 255),
        lineType=cv2.LINE_AA,
        thickness=font_width,
    )
