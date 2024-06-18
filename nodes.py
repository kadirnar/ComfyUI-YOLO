import os
import torch
import numpy as np

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

class LoadUltralytics:
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
        from ultralytics import YOLO


        model_full_path = os.path.join("models/ultralytics", model_path)  # Update with the appropriate directory
        model = YOLO(model_full_path)
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

            },
        }
    RETURN_TYPES = ("ULTRALYTICS_RESULTS","IMAGE","BOXES","MASKS",)
    FUNCTION = "inference"
    CATEGORY = "Ultralytics"

    def inference(self, model, image, conf=0.25, iou=0.7, imgsz=640, device="cuda:0", half=False, augment=False, agnostic_nms=False):
        # video
        if image.shape[0] > 1:
            video_results = []
            video_boxes = []
            video_masks = []
            for torch_img in range(image.shape[0]):
                yolo_image = image[torch_img].permute(2,0,1).unsqueeze(0)
                results = model(yolo_image, conf=conf, iou=iou, imgsz=imgsz, device=device, half=half, augment=augment, agnostic_nms=agnostic_nms)[0]
                video_results.append(results)
                video_boxes.append(results.boxes.xywh)
                video_masks.append(results.masks)
            
            return (video_results, image, video_boxes, video_masks,)


        yolo_image = image.permute(0, 3,1,2)
        results = model(yolo_image, conf=conf, iou=iou, imgsz=imgsz, device=device, half=half, augment=augment, agnostic_nms=agnostic_nms)[0]

        boxes = results.boxes.xywh
        masks = results.masks

        return (results, image, boxes, masks,)


class UltralyticsVisualization:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "results": ("ULTRALYTICS_RESULTS",),
                "line_width": ("INT", {"default": 3}),
                "font_size": ("INT", {"default": 1}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "Ultralytics"

    def visualize(self, results, line_width=3, font_size=1):
        for i, r in enumerate(results):
            # Plot results image
            im_bgr = r.plot(im_gpu=True, line_width=line_width,font_size=font_size)  # BGR-order numpy array

        # Convert the annotated image from numpy array to float32 tensor
        tensor_image = torch.from_numpy(np.array(im_bgr).astype(np.float32) / 255.0)[None,]

        return (tensor_image,)


NODE_CLASS_MAPPINGS = {
    "LoadUltralytics": LoadUltralytics,
    "UltralyticsInference": UltralyticsInference,
    "UltralyticsVisualization": UltralyticsVisualization,
    "ConvertToDict": ConvertToDict,
    "BBoxToXYWH": BBoxToXYWH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadUltralytics": "Load Ultralytics Model",
    "UltralyticsInference": "Ultralytics Inference",
    "UltralyticsVisualization": "Ultralytics Visualization",
    "ConvertToDict": "Convert to Dictionary",
    "BBoxToXYWH": "BBox to XYWH",
}
