# src/models/vit_extractor.py
import torch
from transformers import ViTImageProcessor, ViTModel

class ViTFeatureExtractor:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        print(f"🔄 Loading ViT Feature Extractor ({model_name})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("✅ ViT Feature Extractor Loaded Successfully!")

    def extract_features(self, images):
        """
        استخراج الـ Embeddings (Spatial Features) من الفريمات أو الصور
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # ناخذ الـ [CLS] token representation بحجم (768)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

if __name__ == "__main__":
    extractor = ViTFeatureExtractor()
    dummy_image = torch.randn(3, 224, 224)
    features = extractor.extract_features(dummy_image)
    print(f"✅ ViT Feature Extraction Successful! Embedding Shape: {features.shape}")