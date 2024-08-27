import os
import torch
import shutil
import numpy as np
import onnxruntime as ort
from utils.general import non_max_suppression

class YOLOHandler:
    def __init__(self, config_manager, models_path):
        self.config_manager = config_manager
        self.models_path = models_path
        self.model = None
        self.load_model()

    def load_model(self):
        self.model_existence_check()
        print(f"Loading {self.get_model_name()} with yolov5 for {self.config_manager.get_setting('yolo_mode')} inference.")

        if self.config_manager.get_setting("yolo_mode") == "onnx":
            onnx_provider = {
                "cpu": "CPUExecutionProvider",
                "amd": "DmlExecutionProvider",
                "nvidia": "CUDAExecutionProvider",
            }.get(self.config_manager.get_setting("yolo_device"))

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.model = ort.InferenceSession(
                f"{self.models_path}/{self.get_model_name()}",
                sess_options=so,
                providers=[onnx_provider],
            )
        else:  # PyTorch or TensorRT
            model_path = f"{self.models_path}/{self.get_model_name()}"
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                verbose=False,
                trust_repo=True,
                force_reload=True,
            )

        print("Model loaded.")

    def detect(self, frame):
        if self.config_manager.get_setting("yolo_mode") in ("pytorch", "tensorrt"):
            self.model.conf = self.config_manager.get_setting("confidence") / 100
            self.model.iou = self.config_manager.get_setting("confidence") / 100
            results = self.model(frame, size=[self.config_manager.get_setting("height"), self.config_manager.get_setting("width")])
            
            # YOLOv5 format
            return results.xyxy[0].cpu().numpy()
        elif self.config_manager.get_setting("yolo_mode") == "onnx":
            img = self.preprocess_image(frame)
            outputs = self.model.run(None, {"images": img})
            predictions = torch.from_numpy(outputs[0])
            nms_results = non_max_suppression(
                predictions,
                self.config_manager.get_setting("confidence") / 100,
                self.config_manager.get_setting("confidence") / 100,
                0,
                False,
                max_det=4,
            )
            return nms_results[0].cpu().numpy()

    def preprocess_image(self, frame):
        img = frame.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def export_model(self):
        temp_model_path = f"{os.path.dirname(self.models_path)}/temp.pt"
        shutil.copy(f"{self.models_path}/{self.config_manager.get_setting('yolo_model')}.pt", temp_model_path)
        height = int(self.config_manager.get_setting("height"))
        width = int(self.config_manager.get_setting("width"))
        os.system(
            f"python {os.path.dirname(self.models_path)}/yolov5/export.py --weights {temp_model_path} --include {self.get_mode()} --imgsz {height} {width} --half --device 0"
        )
        os.remove(temp_model_path)
        if self.config_manager.get_setting("yolo_mode") == "tensorrt":
            os.remove(f"{os.path.dirname(self.models_path)}/temp.onnx")
            shutil.move(
                f"{os.path.dirname(self.models_path)}/temp.engine", f"{self.models_path}/{self.get_model_name()}"
            )
        else:
            shutil.move(
                f"{os.path.dirname(self.models_path)}/temp.onnx", f"{self.models_path}/{self.get_model_name()}"
            )
        print("Export complete.")

    def get_model_name(self):
        if self.config_manager.get_setting("yolo_mode") == "pytorch":
            return f"{self.config_manager.get_setting('yolo_model')}.pt"
        return f"{self.config_manager.get_setting('yolo_model')}v5{self.config_manager.get_setting('height')}{self.config_manager.get_setting('width')}Half.{self.get_mode()}"

    def get_mode(self):
        mode = self.config_manager.get_setting("yolo_mode")
        return "engine" if mode == "tensorrt" else mode

    def model_existence_check(self):
        if self.config_manager.get_setting("yolo_mode") != "pytorch" and not os.path.exists(
            f"{self.models_path}/{self.get_model_name()}"
        ):
            print(
                f"Exporting {self.config_manager.get_setting('yolo_model')}.pt to {self.get_model_name()} with yolov5."
            )
            self.export_model()
