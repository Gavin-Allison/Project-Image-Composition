import torch
import cv2
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class Segmenter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)

    def segment(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits, 
            size=image.shape[:2], 
            mode="bilinear", 
            align_corners=False
        )
        mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        return mask

if __name__ == "__main__":
    image = cv2.imread("../test/test_image.png")
    if image is None:
        exit()

    segmenter = Segmenter()
    mask = segmenter.segment(image)

    viz_mask = (mask * (255 // mask.max()) if mask.max() > 0 else mask).astype(np.uint8)
    cv2.imwrite("../test/test_masks.png", viz_mask)