from . infer import InferDetection


class RFSignsDetector(object):
    def __init__(self, model_path):
        
        self.infer = InferDetection.from_file(
            model_filepath=model_path,
            conf_threshold=0.5,
            nms_threshold=0.3,
            # use_half_precision=True,  # use it only for GPU device!
        )
        self.label_names = self.infer.get_labels()
    
    def find_signs(self, image):
        # Must be RGB image!
        bboxes, label_ids, scores = self.infer.infer_image(image)
        labels = self.infer.map_labels(label_ids)
        return bboxes, labels, scores
    
