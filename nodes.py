
import os
import torch
import numpy as np
from ultralytics import YOLO, SAM
import requests
import json
import comfy
from torchvision import transforms
import torch.nn.functional as F
from nodes import MAX_RESOLUTION
import torchvision
from PIL import Image, ImageDraw
import cv2


coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class BBoxVisNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BOXES",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "color": ("STRING", {"default": "red"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_bbox"

    CATEGORY = "Ultralytics/Utils"


    def draw_bbox(self, image, bboxes, index, color):
        img = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_image)

        bbox = bboxes[index]
        x, y, w, h = bbox
        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        tensor_image = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)
        tensor_image = tensor_image.unsqueeze(0)
        
        return (tensor_image,)

class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Width", "Height")
    FUNCTION = "get_image_size"

    CATEGORY = "Ultralytics/Utils"

    def get_image_size(self, image):
        return (image.shape[1], image.shape[2],)

class ImageResizeAdvanced:
    # https://github.com/cubiq/ComfyUI_essentials/blob/main/image.py
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "Ultralytics/Utils"

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = ow

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]

        return(outputs, outputs.shape[2], outputs.shape[1],)

class CocoToNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coco_label": (
                    coco_classes,
                    {"default": "person"},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "map_class"
    CATEGORY = "Ultralytics/Utils"

    def map_class(self, coco_label):
        class_num = str(coco_classes.index(coco_label))

        return (class_num,)

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
                        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
                        
                    ],
                ),
            },
        }

    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Ultralytics/Model"

    def coco_to_labels(self, coco):
        labels = []
        for category in coco["categories"]:
            labels.append(category["name"])
        return labels

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

    CATEGORY = "Ultralytics/Model"
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
                        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
                        "mobile_sam.pt"
                    ],
                ),
            },
        }

    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Ultralytics/Model"

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

class BBoxToCoco:
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

    CATEGORY = "Ultralytics/Utils"

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
    RETURN_NAMES = ("StrBox", "BOXES","X_coord", "Y_coord", "Width", "Height",)
    FUNCTION = "bbox_to_xywh"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Utils"

    def bbox_to_xywh(self, index, bbox):
        bbox = bbox[index]

        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        fullstr = f"x: {x}, y: {y}, w: {w}, h: {h}"

        return (fullstr,bbox, x,y,w,h,)

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

    CATEGORY = "Ultralytics/Utils"

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
    CATEGORY = "Ultralytics/Inference"

    def inference(self, model, image, conf=0.25, iou=0.7, height=640, width=640, device="cuda:0", half=False, augment=False, agnostic_nms=False, classes=None):
        if classes == "None":
            class_list = None
        else:
            class_list = [int(cls.strip()) for cls in classes.split(',')]

        if image.shape[0] > 1:
            batch_size = image.shape[0]
            results = []
            for i in range(batch_size):
                yolo_image = image[i].unsqueeze(0).permute(0, 3, 1, 2)
                result = model.predict(yolo_image, conf=conf, iou=iou, imgsz=(height, width), device=device, half=half, augment=augment, agnostic_nms=agnostic_nms, classes=class_list)
                results.append(result)

            boxes = [result[0].boxes.xywh for result in results]
            masks = [result[0].masks for result in results]
            probs = [result[0].probs for result in results]
            keypoints = [result[0].keypoints for result in results]
            obb = [result[0].obb for result in results]

        else:
            yolo_image = image.permute(0, 3, 1, 2)
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
                "sam": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "Ultralytics/Vis"

    def visualize(self, image, results, line_width=3, font_size=1, sam=False):
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
                if sam == True:
                    im_bgr = r.plot(line_width=line_width, font_size=font_size)  # BGR-order numpy array

                else:
                    im_bgr = r.plot(im_gpu=True, line_width=line_width, font_size=font_size)  # BGR-order numpy array
                annotated_frames.append(im_bgr)

            tensor_image = torch.stack([torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in annotated_frames])

        return (tensor_image,)

class ViewText:
    # https://github.com/gokayfem/ComfyUI_VLM_nodes/blob/main/nodes/simpletext.py
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "view_text"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Utils"

    def view_text(self, text):
        # Parse the combined JSON string
        return {"ui": {"text": text}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "UltralyticsModelLoader": UltralyticsModelLoader,
    "UltralyticsInference": UltralyticsInference,
    "UltralyticsVisualization": UltralyticsVisualization,
    "ConvertToDict": ConvertToDict,
    "BBoxToXYWH": BBoxToXYWH,
    "BBoxToCoco": BBoxToCoco,
    "CustomUltralyticsModelLoader": CustomUltralyticsModelLoader,
    "CocoToNumber": CocoToNumber,
    "GetImageSize": GetImageSize,
    "ImageResizeAdvanced": ImageResizeAdvanced,
    "BBoxVisNode": BBoxVisNode,
    "ViewText": ViewText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltralyticsModelLoader": "Ultralytics Model Loader",
    "UltralyticsInference": "Ultralytics Inference",
    "UltralyticsVisualization": "Ultralytics Visualization",
    "ConvertToDict": "Convert to Dictionary",
    "BBoxToXYWH": "BBox to XYWH",
    "BBoxToCoco": "BBox to Coco",
    "CustomUltralyticsModelLoader": "Custom Ultralytics Model Loader",
    "CocoToNumber": "Coco to Number",
    "GetImageSize": "Get Image Size",
    "ImageResizeAdvanced": "Image Resize Advanced",
    "BBoxVisNode": "BBox Visualization",
    "ViewText": "View Text",
}
