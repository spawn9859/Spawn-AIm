import os
import shutil
import torch
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import ops
from print_utils import pr_blue, pr_green, pr_cyan

def get_mode(settings):
    if settings['yolo_mode'] == "pytorch":
        return "engine"
    elif settings['yolo_mode'] == "onnx":
        return "engine"
    elif settings['yolo_mode'] == "tensorrt":
        return "engine"
    return None

def get_model_name(settings):
    if settings['yolo_mode'] == "pytorch":
        return f"{settings['yolo_model']}.pt"
    else:
        return f"{settings['yolo_model']}{settings['yolo_version']}{settings['height']}{settings['width']}Half.{get_mode(settings)}"

def model_existence_check(settings, models_path, script_directory):
    if not settings['yolo_mode'] == "pytorch":
        if not os.path.exists(f"{models_path}/{get_model_name(settings)}"):
            pr_cyan(f"Exporting {settings['yolo_model']}.pt to {get_model_name(settings)} with yolo{settings['yolo_version']}.")
            export_model(settings, models_path, script_directory)

def export_model(settings, models_path, script_directory):
    shutil.copy(f"{models_path}/{settings['yolo_model']}.pt", f"{script_directory}/temp.pt")
    if settings['yolo_version'] == "v8":
        x_model = YOLO(f"{script_directory}/temp.pt")
        x_model.export(format=get_mode(settings), imgsz=[int(settings['height']), int(settings['width'])], half=True, device=0)
    elif settings['yolo_version'] == "v5":
        os.system(f"python {script_directory}/yolov5/export.py --weights {script_directory}/temp.pt --include {get_mode(settings)} --imgsz {int(settings['height'])} {int(settings['width'])} --half --device 0")
    os.remove(f"{script_directory}/temp.pt")
    if settings['yolo_mode'] == "tensorrt":
        os.remove(f"{script_directory}/temp.onnx")
        shutil.move(f"{script_directory}/temp.engine", f"{models_path}/{get_model_name(settings)}")
    else:
        shutil.move(f"{script_directory}/temp.onnx", f"{models_path}/{get_model_name(settings)}")
    pr_green("Export complete.")

def load_model(settings, models_path, script_directory):
    global model
    model_existence_check(settings, models_path, script_directory)
    pr_blue(f"Loading {get_model_name(settings)} with yolo{settings['yolo_version']} for {settings['yolo_mode']} inference.")
    pr_blue(f"Model path: {models_path}/{get_model_name(settings)}")

    if not settings['yolo_mode'] == "onnx":
        if settings['yolo_version'] == "v8":
            model = YOLO(f"{models_path}/{get_model_name(settings)}", task="detect")
        elif settings['yolo_version'] == "v5":
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=f"{models_path}/{get_model_name(settings)}", verbose=False, trust_repo=True, force_reload=True)
    else:
        onnx_provider = ""
        if settings['yolo_device'] == "cpu":
            onnx_provider = "CPUExecutionProvider"
        elif settings['yolo_device'] == "amd":
            onnx_provider = "DmlExecutionProvider"
        elif settings['yolo_device'] == "nvidia":
            onnx_provider = "CUDAExecutionProvider"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = ort.InferenceSession(f"{models_path}/{get_model_name(settings)}", sess_options=so, providers=[onnx_provider])

    if model is None:
        raise ValueError(f"Failed to load model: {get_model_name(settings)}")
    else:
        pr_green("Model loaded successfully.")
    return model
