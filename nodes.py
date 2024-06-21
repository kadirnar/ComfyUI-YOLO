import os
import torch
import numpy as np
from ultralytics import YOLO
import requests
import json
import os
import urllib.request
import comfy
import os
import urllib.request

class CustomUltralyticsModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        models_dir = "models/ultralytics"  # Update with the appropriate directory
        files = []
        for root, dirs, filenames in os.walk(models_dir):
            for filename in filenames:
                if filename.endswith(".pt"):
                    relative_path = os.path.relpath(os.path.join(root, filename), models_dir)
                    files.append(relative_path)
        return {
            "required": {
                "model_path": (sorted(files), {"model_upload": True})
            }
        }

    CATEGORY = "Ultralytics"
    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_path):
        model_full_path = os.path.join("models/ultralytics", model_path)  # Update with the appropriate directory
        model = YOLO(model_full_path)
        return (model,)



class UltralyticsModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_name": (
                    [
                        "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
                        "yolov5n6u.pt", "yolov5s6u.pt", "yolov5m6u.pt", "yolov5l6u.pt", "yolov5x6u.pt",
                        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                        "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
                        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt"
                    ],
                ),
            },
        }

    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Model Loading"

    def __init__(self):
        self.loaded_models = {}

    def load_model(self, model_name=None):
        if model_name is None:
            model_name = "yolov8s.pt"  # Default model name if not provided

        if model_name in self.loaded_models:
            print(f"Model {model_name} already loaded. Returning cached model.")
            return (self.loaded_models[model_name],)

        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}"

        # Create a "models/ultralytics" directory if it doesn't exist
        os.makedirs(os.path.join("models", "ultralytics"), exist_ok=True)

        model_path = os.path.join("models", "ultralytics", model_name)

        # Check if the model file already exists
        if os.path.exists(model_path):
            print(f"Model {model_name} already downloaded. Loading model.")
        else:
            print(f"Downloading model {model_name}...")
            response = requests.get(model_url)
            response.raise_for_status()  # Raise an exception if the download fails

            with open(model_path, "wb") as file:
                file.write(response.content)

            print(f"Model {model_name} downloaded successfully.")

        model = YOLO(model_path)
        self.loaded_models[model_name] = model
        return (model,)


class BBoxToCOCO:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "results": ("ULTRALYTICS_RESULTS",),
                "bbox": ("BOXES",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coco_json",)
    FUNCTION = "bbox_to_xywh"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Text"

    def bbox_to_xywh(self, results, bbox):
        coco_data = {
            "categories": [],
            "images": [],
            "annotations": [],
        }

        annotation_id = 1
        category_names = results[0].names

        if isinstance(bbox, list):
            for frame_idx, bbox_frame in enumerate(bbox):
                image_id = frame_idx + 1
                image_width, image_height = results[frame_idx].boxes.orig_shape[1], results[frame_idx].boxes.orig_shape[0]
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": f"{image_id:04d}.jpg",
                    "height": image_height,
                    "width": image_width,
                })

                for bbox_single, cls_single in zip(bbox_frame, results[frame_idx].boxes.cls):
                    x = float(bbox_single[0])
                    y = float(bbox_single[1])
                    w = float(bbox_single[2])
                    h = float(bbox_single[3])
                    category_id = int(cls_single.item()) + 1

                    if category_id not in [cat["id"] for cat in coco_data["categories"]]:
                        coco_data["categories"].append({
                            "id": category_id,
                            "name": category_names[category_id - 1],
                            "supercategory": "none"
                        })

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    annotation_id += 1
        else:
            image_id = 1
            image_width, image_height = results[0].boxes.orig_shape[1], results[0].boxes.orig_shape[0]
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{image_id:04d}.jpg",
                "height": image_height,
                "width": image_width,
            })

            for bbox_single, cls_single in zip(bbox, results[0].boxes.cls):
                if bbox_single.dim() == 0:
                    x = float(bbox_single.item())
                    y = float(bbox_single.item())
                    w = float(bbox_single.item())
                    h = float(bbox_single.item())
                else:
                    x = float(bbox_single[0])
                    y = float(bbox_single[1])
                    w = float(bbox_single[2])
                    h = float(bbox_single[3])

                category_id = int(cls_single.item()) + 1

                if category_id not in [cat["id"] for cat in coco_data["categories"]]:
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": category_names[category_id - 1],
                        "supercategory": "none"
                    })

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1

        coco_json = json.dumps(coco_data, indent=2)
        return (coco_json,)



class BBoxToXYWH:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0}),
                "bbox": ("BOXES", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("StrBox", "X_coord", "Y_coord", "Width", "Height",)
    FUNCTION = "bbox_to_xywh"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Text"

    def bbox_to_xywh(self, index, bbox):
        bbox = bbox[index]

        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        fullstr = f"x: {x}, y: {y}, w: {w}, h: {h}"

        return (fullstr,x,y,w,h,)


class ConvertToDict:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "bbox": ("BOXES", {"default": None}),
                "mask": ("MASKS", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_to_dict"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Text"

    def convert_to_dict(self, bbox=None, mask=None):
        output = {"objects": []}

        if bbox is not None:
            for obj_bbox in bbox:
                bbox_dict = {
                    "x": obj_bbox[0].item(),
                    "y": obj_bbox[1].item(),
                    "width": obj_bbox[2].item(),
                    "height": obj_bbox[3].item()
                }
                output["objects"].append({"bbox": bbox_dict})

        if mask is not None:
            for obj_mask in mask:
                mask_dict = {
                    "shape": obj_mask.shape,
                    "data": obj_mask.tolist()
                }
                if len(output["objects"]) > len(mask):
                    output["objects"].append({"mask": mask_dict})
                else:
                    output["objects"][-1]["mask"] = mask_dict

        if not output["objects"]:
            output = {"message": "No input provided"}

        import json
        output_str = json.dumps(output, indent=2)

        return {"ui": {"text": output_str}, "result": (output_str,)}


class UltralyticsInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("ULTRALYTICS_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "conf": ("FLOAT", {"default": 0.25, "min": 0, "max": 1, "step": 0.01}),
                "iou": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.01}),
                "height": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 32}),
                "width": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 32}),
                "device":(["cuda:0", "cpu"], ),
                "half": ("BOOLEAN", {"default": False}),
                "augment": ("BOOLEAN", {"default": False}),
                "agnostic_nms": ("BOOLEAN", {"default": False}),
                "classes": ("STRING", {"default": "None"}),

            },
        }
    RETURN_TYPES = ("ULTRALYTICS_RESULTS","IMAGE", "BOXES", "MASKS", "PROBABILITIES", "KEYPOINTS", "OBB",)
    FUNCTION = "inference"
    CATEGORY = "Ultralytics"

    def inference(self, model, image, conf=0.25, iou=0.7, height=640, width=640, device="cuda:0", half=False, augment=False, agnostic_nms=False, classes=None):
        if classes == "None":
            class_list = None
        else:
            class_list = [int(cls.strip()) for cls in classes.split(',')]

        if image.shape[0] > 1:
            batch_size = image.shape[0]
            results = []
            for i in range(batch_size):
                yolo_image = torch.nn.functional.interpolate(image[i].unsqueeze(0).permute(0, 3, 1, 2), size=(height, width), mode='bilinear', align_corners=False)
                result = model.predict(yolo_image, conf=conf, iou=iou, imgsz=(height, width), device=device, half=half, augment=augment, agnostic_nms=agnostic_nms, classes=class_list)
                results.append(result)

            boxes = [result[0].boxes.xywh for result in results]
            masks = [result[0].masks for result in results]
            probs = [result[0].probs for result in results]
            keypoints = [result[0].keypoints for result in results]
            obb = [result[0].obb for result in results]

        else:
            yolo_image = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), size=(height, width), mode='bilinear', align_corners=False)
            results = model.predict(yolo_image, conf=conf, iou=iou, imgsz=(height,width), device=device, half=half, augment=augment, agnostic_nms=agnostic_nms, classes=class_list)

            boxes = results[0].boxes.xywh
            masks = results[0].masks
            probs = results[0].probs
            keypoints = results[0].keypoints
            obb = results[0].obb            

        return (results, image, boxes, masks, probs, keypoints, obb,)


class UltralyticsVisualization:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "results": ("ULTRALYTICS_RESULTS",),
                "image": ("IMAGE",),
                "line_width": ("INT", {"default": 3}),
                "font_size": ("INT", {"default": 1}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "Ultralytics"

    def visualize(self, image, results, line_width=3, font_size=1):
        if image.shape[0] > 1:
            batch_size = image.shape[0]
            annotated_frames = []
            for result in results:
                for r in result:
                    im_bgr = r.plot(im_gpu=True, line_width=line_width, font_size=font_size)
                    annotated_frames.append(im_bgr)

            tensor_image = torch.stack([torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in annotated_frames])

        else:
            annotated_frames = []
            for r in results:
                im_bgr = r.plot(im_gpu=True, line_width=line_width, font_size=font_size)  # BGR-order numpy array
                annotated_frames.append(im_bgr)

            tensor_image = torch.stack([torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in annotated_frames])

        return (tensor_image,)


NODE_CLASS_MAPPINGS = {
    "UltralyticsModelLoader": UltralyticsModelLoader,
    "UltralyticsInference": UltralyticsInference,
    "UltralyticsVisualization": UltralyticsVisualization,
    "ConvertToDict": ConvertToDict,
    "BBoxToXYWH": BBoxToXYWH,
    "BBoxToCOCO": BBoxToCOCO,
    "CustomUltralyticsModelLoader": CustomUltralyticsModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltralyticsModelLoader": "Ultralytics Model Loader",
    "UltralyticsInference": "Ultralytics Inference",
    "UltralyticsVisualization": "Ultralytics Visualization",
    "ConvertToDict": "Convert to Dictionary",
    "BBoxToXYWH": "BBox to XYWH",
    "BBoxToCOCO": "BBox to COCO",
    "CustomUltralyticsModelLoader": "Custom Ultralytics Model Loader",
}
