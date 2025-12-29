import torch
import clip
from PIL import Image


class CLIPEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def embed(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_input)

        embedding = embedding.squeeze(0)
        embedding = embedding / embedding.norm()
        return embedding
    
if __name__ == "__main__":
    embedder = CLIPEmbedder()

    v1 = embedder.embed("data/images/cat.jpg")
    v2 = embedder.embed("data/images/cat_backrem.jpg")

    sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0)
    print("Similarity:", sim.item())