from __future__ import annotations

import torch
import numpy as np
import logging

from dcvt.detection.utils.bbox import diou_xywh_torch
from dcvt.detection.utils.nms import TorchNMS
from dcvt.detection.models import construct_model
from dcvt.common.utils.torch import set_half_precision

from dcvt.detection.preprocessing import deserialize_preprocessing

from concurrent.futures import ProcessPoolExecutor as PoolExecutor


class InferDetection():
    def __init__(
        self, model, device=None, nms_threshold=0.6, conf_threshold=0.4, 
        use_soft_nms=False, use_half_precision=False, n_processes=1
    ):
        self.model = model
        self.config = model.config
        
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f'Loading model with config: {self.config}')

        self.n_processes = n_processes
        self.model_hw = self.config['infer_sz_hw']
        self.use_half_precision = use_half_precision
        self.use_soft_nms = use_soft_nms

        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.model.to(self.device)
        self.model.eval()

        if self.use_half_precision:
            set_half_precision(self.model)

        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.labels = self.config['labels']

        self._size_tnsr = torch.FloatTensor([
            self.model_hw[1],
            self.model_hw[0],
            self.model_hw[1],
            self.model_hw[0],
        ]).view(1, 1, 4).to(self.device)

        self.preproc_ops = deserialize_preprocessing(self.config['preprocessing'])
        self.nms = TorchNMS(
            iou=diou_xywh_torch,
            iou_threshold=nms_threshold
        )

    def map_labels(self, label_ids: list) -> list:
        """Map predicted IDs into string class labels

        Parameters
        ----------
        label_ids : list
            Predicted class indexes

        Returns
        -------
        list
            Predicted classes
        """
        return [self.labels[id_] for id_ in label_ids]

    @property
    def name(self) -> str:
        """Model name

        Returns
        -------
        str
            Name of internal model
        """
        return self.model.__name__
        
    @classmethod
    def from_config(cls, model_config: dict, **kwargs) -> InferDetection:
        """Create model from config, it creates another instance so to sync use update_model_state()

        Args:
            model_config (dict): model configuration

        Returns:
            InferDetection: class for inference
        """
        print(model_config)
        model = construct_model(model_config, inference=True)
        return cls(model=model, **kwargs)

    @classmethod
    def from_file(cls, model_filepath: str, device=None, **kwargs) -> InferDetection:
        """Create model from serialized file, it creates another instance so to sync use update_model_state()

        Args:
            model_filepath (str): model path to file in filesystem

        Returns:
            InferDetection: class for inference
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        loaded_data = torch.load(model_filepath, map_location=device)
        model_config = loaded_data['model_config']
        model_state = loaded_data['model_state']
        
        model = cls.from_config(
            model_config=model_config,
            device=device,
            **kwargs
        )
        model.update_model_state(model_state)
        return model

    def get_labels(self) -> list[str]:
        """Get list of labels that model was trained on

        Returns
        -------
        list[str]
            Trained labels
        """
        return self.labels

    def update_model_state(self, model_state: dict):
        """Update state (weights) of model

        Parameters
        ----------
        model_state : dict
            Weights
        """
        self.model.load_state_dict(model_state)

    def infer_image(self, image: np.array) -> np.array:
        """Perform prediction on single image

        Parameters
        ----------
        image : np.array
            Input image (RGB format)

        Returns
        -------
        np.array
            Predictions
        """
        return self.infer_batch([image])[0]

    def infer_batch(self, imgs_list: list[np.array]) -> list:
        """Perform prediction on multiple images

        Parameters
        ----------
        imgs_list : list[np.array]
            Images list (RGB format)

        Returns
        -------
        list
            List of predictions
        """
        batch_tensor = []
        
        # Letterboxing can be optimized with applying matrix operations (scale, pad)
        preproc_data = []
        for img in imgs_list:
            data = {}
            for op in self.preproc_ops:
                img, _ = op.transform(img, data=data)

            preproc_data.append(data)
            batch_tensor.append(img)

        batch_tensor = torch.stack(batch_tensor, axis=0)
        
        # NOTE - here we don`t apply shift and scale to all bboxes at once - just to demonstrate
        with torch.no_grad():
            batch_tensor = batch_tensor.to(self.device)
            if self.use_half_precision:
                batch_tensor = batch_tensor.half()

            outputs = self.model(batch_tensor)
            outputs[..., :4] *= self._size_tnsr
            
            if self.use_half_precision:
                outputs = outputs.float()

            outputs = outputs.cpu()

        # Go through batches
        result_list = []

        if self.n_processes > 1:
            with PoolExecutor(self.n_processes) as ex:
                futures = []
                for i, output in enumerate(outputs):
                    # Normalized
                    preds = output[output[..., 4] > self.conf_threshold]
                    fut = ex.submit(
                        self._process_prediction, 
                        preds, preproc_data[i], self.nms, self.preproc_ops
                    )
                    futures.append(fut)

                for fut in futures:
                    bboxes, labels, scores = fut.result()
                    # Tuple of three components
                    result_list.append((
                        bboxes,
                        labels,
                        scores,
                    ))
        else:
            for i, output in enumerate(outputs):
                preds = output[output[..., 4] > self.conf_threshold]
                # print(f'Received {preds.shape} predictions')

                bboxes, labels, scores = self._process_prediction(
                    preds, preproc_data[i], self.nms, self.preproc_ops
                )
                result_list.append((
                    bboxes,
                    labels,
                    scores,
                ))

        return result_list

    @staticmethod
    def _process_prediction(preds, preproc_data, nms, preproc_ops):
        if preds.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        keep = nms.exec(preds)
        
        preds = preds[keep] #.cpu()
        for op in reversed(preproc_ops):
            preds = op.inverse_transform(preds=preds, data=preproc_data)

        bboxes = preds[:, :4]
        labels = preds[:, 5].astype(int)
        scores = preds[:, 4]

        return bboxes, labels, scores
