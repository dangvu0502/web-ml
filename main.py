import json
from PIL import Image
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from transformers import CLIPProcessor, CLIPModel
import unittest
import numpy as np

json_path = 'template_data.json'
image_path = 'templates/'

with open(json_path, 'r') as f:
    input_data = []
    for line in f:
        obj = json.loads(line)
        input_data.append(obj)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Define a custom dataset
class ImageTitleDataset():
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

# Use your own data
list_image_path = []
list_txt = []
for item in input_data:
    img_path = image_path + item['image_path'].split('/')[-1]
    caption = item['captions'][:40]
    list_image_path.append(img_path)
    list_txt.append(caption)

dataset = ImageTitleDataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size=1000, shuffle=True) # Define your own dataloader

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model (commented out for brevity)
# num_epochs = 2
# for epoch in range(num_epochs):
#     pbar = tqdm(train_dataloader, total=len(train_dataloader))
#     for batch in pbar:
#         optimizer.zero_grad()
#         images, texts = batch
#         images = images.to(device)
#         texts = texts.to(device)
#         logits_per_image, logits_per_text = model(images, texts)
#         ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
#         total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
#         total_loss.backward()
#         if device == "cpu":
#             optimizer.step()
#         else:
#             convert_models_to_fp32(model)
#             optimizer.step()
#             clip.model.convert_weights(model)
#         pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

print("Training complete.")

stored_images = np.empty(0)

# Testing the model with text input
def find_similar_images(text_input, image_dir):
    # Load and preprocess all images in the directory
    global stored_images
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]
    images = []
    valid_image_paths = []
    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            images.append(image)
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    if not images:
        print("No valid images found for testing.")
        return []

    images = torch.cat(images)  # Stack all image tensors

    # Tokenize and encode the text input
    text = clip.tokenize([text_input]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Flatten each image feature tensor and concatenate them into a 1D array
        flattened_features = np.concatenate([img_feat.flatten().cpu().numpy() for img_feat in image_features])
        stored_images = np.concatenate([stored_images, flattened_features]) if len(stored_images) else flattened_features
        
        # Calculate similarity and sort images by similarity
        similarities = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        sorted_indices = similarities.argsort(descending=True).squeeze().tolist()

    sorted_image_paths = [valid_image_paths[idx] for idx in sorted_indices]
    return sorted_image_paths

# Example text input
text_input = "a raised part of the earth's surface, much larger than a hill, the top of which might be covered in snow"
similar_images = find_similar_images(text_input, image_path)

file_name = 'stored_images.bin'
# Save the numpy array to a .npy file
stored_images.astype(np.float32,).tofile(file_name)

print("Stored images have been saved to "  + file_name)

# Function to load the stored images from the .npy file
def load_stored_images(npy_file_path):
    return np.fromfile(npy_file_path , dtype=np.float32) 

# Usage example
loaded_images = load_stored_images(file_name)
print(f"Loaded {len(loaded_images)} stored image feature sets from " + file_name)

# Print the first few elements to check content
print("First few elements in " + file_name)
print(loaded_images[:10])  # Display the first 10 elements for a brief check
