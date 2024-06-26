import os
import torch
import shutil
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from utils.general import non_max_suppression
from ultralytics.utils import ops

class YOLOHandler:
    def __init__(self, config_manager, models_path):
        self.config_manager = config_manager
        self.models_path = models_path
        self.model = None
        self.load_model()

    def load_model(self):
        self.model_existence_check()
        print(f"Loading {self.get_model_name()} with yolo{self.config_manager.get_setting('yolo_version')} for {self.config_manager.get_setting('yolo_mode')} inference.")

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
            if self.config_manager.get_setting("yolo_version") == "v8":
                self.model = YOLO(model_path, task="detect")
            elif self.config_manager.get_setting("yolo_version") == "v5":
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
            if self.config_manager.get_setting("yolo_version") == "v5":
                self.model.conf = self.config_manager.get_setting("confidence") / 100
                self.model.iou = self.config_manager.get_setting("confidence") / 100
                return self.model(frame, size=[self.config_manager.get_setting("height"), self.config_manager.get_setting("width")])
            elif self.config_manager.get_setting("yolo_version") == "v8":
                return self.model.predict(
                frame,
                verbose=False,
                conf=self.config_manager.get_setting("confidence") / 100,
                iou=self.config_manager.get_setting("confidence") / 100,
                half=False,
                imgsz=[self.config_manager.get_setting("height"), self.config_manager.get_setting("width")],
            )
        elif self.config_manager.get_setting("yolo_mode") == "onnx":
            # ... (ONNX preprocessing code remains the same)
            outputs = self.model.run(None, {"images": frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame})
            predictions = torch.from_numpy(outputs[0])
            if self.config_manager.get_setting("yolo_version") == "v5":
                return non_max_suppression(
                    predictions,
                    self.config_manager.get_setting("confidence") / 100,
                    self.config_manager.get_setting("confidence") / 100,
                    0,
                    False,
                    max_det=4,
                )
            elif self.config_manager.get_setting("yolo_version") == "v8":
                return ops.non_max_suppression(
                    predictions,
                    self.config_manager.get_setting("confidence") / 100,
                    self.config_manager.get_setting("confidence") / 100,
                    0,
                    False,
                    max_det=4,
                )

    def export_model(self):
        temp_model_path = f"{os.path.dirname(self.models_path)}/temp.pt"
        shutil.copy(f"{self.models_path}/{self.config_manager.get_setting('yolo_model')}.pt", temp_model_path)
        height = int(self.config_manager.get_setting("height"))
        width = int(self.config_manager.get_setting("width"))
        if self.config_manager.get_setting("yolo_version") == "v8":
            x_model = YOLO(temp_model_path)
            x_model.export(
                format=self.get_mode(), imgsz=[height, width], half=True, device=0
            )
        elif self.config_manager.get_setting("yolo_version") == "v5":
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
        return f"{self.config_manager.get_setting('yolo_model')}{self.config_manager.get_setting('yolo_version')}{self.config_manager.get_setting('height')}{self.config_manager.get_setting('width')}Half.{self.get_mode()}"

    def get_mode(self):
        mode = self.config_manager.get_setting("yolo_mode")
        return "engine" if mode in ("pytorch", "onnx", "tensorrt") else None

    def model_existence_check(self):
        if self.config_manager.get_setting("yolo_mode") != "pytorch" and not os.path.exists(
            f"{self.models_path}/{self.get_model_name()}"
        ):
            print(
                f"Exporting {self.config_manager.get_setting('yolo_model')}.pt to {self.get_model_name()} with yolo{self.config_manager.get_setting('yolo_version')}."
            )
            self.export_model()
