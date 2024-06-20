import os
import torch
import numpy as np
from ultralytics import YOLO
import requests
import json

class UltralyticsModelDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_name": (
                    [
                        "FastSAM-s.pt", "FastSAM-x.pt", "mobile_sam.pt", "rtdetr-l.pt", "rtdetr-x.pt",
                        "sam_b.pt", "sam_l.pt", "yolov10b.pt", "yolov10l.pt", "yolov10m.pt",
                        "yolov10n.pt", "yolov10s.pt", "yolov10x.pt",
                        "yolov5l6u.pt", "yolov5lu.pt", "yolov5m6u.pt", "yolov5mu.pt",
                        "yolov5n6u.pt", "yolov5nu.pt", "yolov5s6u.pt", "yolov5su.pt", "yolov5x6u.pt",
                        "yolov5xu.pt", "yolov8l-cls.pt", "yolov8l-e2e.pt", "yolov8l-human.pt", "yolov8l-obb.pt",
                        "yolov8l-oiv7.pt", "yolov8l-pose.pt", "yolov8l-seg.pt", "yolov8l-v8loader.pt", "yolov8l-world-cc3m.pt",
                        "yolov8l-world.pt", "yolov8l-worldv2-cc3m.pt", "yolov8l-worldv2.pt", "yolov8l.pt", "yolov8m-cls.pt",
                        "yolov8m-human.pt", "yolov8m-obb.pt", "yolov8m-oiv7.pt", "yolov8m-pose.pt", "yolov8m-seg.pt",
                        "yolov8m-v8loader.pt", "yolov8m-world.pt", "yolov8m-worldv2.pt", "yolov8m.pt", "yolov8n-cls.pt",
                        "yolov8n-e2e.pt", "yolov8n-human.pt", "yolov8n-obb.pt", "yolov8n-oiv7.pt", "yolov8n-pose.pt",
                        "yolov8n-seg.pt", "yolov8n-v8loader.pt", "yolov8n.pt", "yolov8s-cls.pt", "yolov8s-e2e.pt",
                        "yolov8s-human.pt", "yolov8s-obb.pt", "yolov8s-oiv7.pt", "yolov8s-pose.pt", "yolov8s-seg.pt",
                        "yolov8s-v8loader.pt", "yolov8s-world.pt", "yolov8s-worldv2.pt", "yolov8s.pt", "yolov8x-cls.pt",
                        "yolov8x-e2e.pt", "yolov8x-human.pt", "yolov8x-obb.pt", "yolov8x-oiv7.pt", "yolov8x-pose-p6.pt",
                        "yolov8x-pose.pt", "yolov8x-seg.pt", "yolov8x-v8loader.pt", "yolov8x-world.pt", "yolov8x-worldv2.pt",
                        "yolov8x.pt", "yolov8x6-500.pt", "yolov8x6-oiv7.pt", "yolov8x6.pt", "yolov9c-seg.pt",
                        "yolov9c.pt", "yolov9e-seg.pt", "yolov9e.pt", "yolov9m.pt", "yolov9s.pt",
                        "yolov9t.pt", "yolo_nas_l.pt", "yolo_nas_m.pt", "yolo_nas_s.pt"
                    ],
                ),
            },
        }

    RETURN_TYPES = ("MODEL_PATH",)
    FUNCTION = "download_model"
    CATEGORY = "Model"

    def __init__(self):
        self.loaded_models = set()

    def download_model(self, model_name=None):
        if model_name is None:
            model_name = "yolov8s.pt"  # Default model name if not provided

        if model_name in self.loaded_models:
            print(f"Model {model_name} already loaded. Skipping download.")
            return (os.path.join("models", "ultralytics", model_name),)

        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}"

        # Create a "models/ultralytics" directory if it doesn't exist
        os.makedirs(os.path.join("models", "ultralytics"), exist_ok=True)

        model_path = os.path.join("models", "ultralytics", model_name)

        # Check if the model file already exists
        if os.path.exists(model_path):
            print(f"Model {model_name} already downloaded. Skipping download.")
        else:
            print(f"Downloading model {model_name}...")
            response = requests.get(model_url)
            response.raise_for_status()  # Raise an exception if the download fails

            with open(model_path, "wb") as file:
                file.write(response.content)

            print(f"Model {model_name} downloaded successfully.")

        self.loaded_models.add(model_name)
        return (model_path,)

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


class LoadUltralyticsModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("MODEL_PATH",),
            },
        }
    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Model Loading"

    def load_model(self, model_path):
        model = YOLO(model_path)
        return (model,)


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
                "imgsz": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 32}),
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

    def inference(self, model, image, conf=0.25, iou=0.7, imgsz=640, device="cuda:0", half=False, augment=False, agnostic_nms=False, classes=["0"]):
        if classes == "None":
            class_list = None
        else:
            class_list = [int(cls.strip()) for cls in classes.split(',')]

        if image.shape[0] > 1:
            batch_size = image.shape[0]
            results = []
            for i in range(batch_size):
                yolo_image = image[i].unsqueeze(0).permute(0, 3, 1, 2)
                result = model.predict(yolo_image, conf=conf, iou=iou, imgsz=imgsz, device=device, half=half, augment=augment, agnostic_nms=agnostic_nms, classes=class_list)
                results.append(result)

        else:
            yolo_image = image.permute(0, 3,1,2)
            results = model.predict(yolo_image, conf=conf, iou=iou, imgsz=imgsz, device=device, half=half, augment=augment, agnostic_nms=agnostic_nms, classes=class_list)

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
    "LoadUltralyticsModel": LoadUltralyticsModel,
    "UltralyticsInference": UltralyticsInference,
    "UltralyticsVisualization": UltralyticsVisualization,
    "ConvertToDict": ConvertToDict,
    "BBoxToXYWH": BBoxToXYWH,
    "UltralyticsModelDownloader": UltralyticsModelDownloader,
    "BBoxToCOCO": BBoxToCOCO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadUltralyticsModel": "Load Ultralytics Model",
    "UltralyticsInference": "Ultralytics Inference",
    "UltralyticsVisualization": "Ultralytics Visualization",
    "ConvertToDict": "Convert to Dictionary",
    "BBoxToXYWH": "BBox to XYWH",
    "UltralyticsModelDownloader": "Ultralytics Model Downloader",
    "BBoxToCOCO": "BBox to COCO",
}
