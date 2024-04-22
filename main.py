import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, size=512, scale=None):
    image = Image.open(image_path).convert('RGB')
    if scale:
        size = int(scale * min(image.size))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def tensor_to_image(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image

def save_image(image_tensor, output_path):
    image = tensor_to_image(image_tensor)
    image.save(output_path)

def main(content_image_path, style_image_path, output_path):
    content_image = load_image(content_image_path)
    style_image = load_image(style_image_path, scale=0.5)

    model = models.vgg19(pretrained=True).features
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.MaxPool2d):
            model[int(name)] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Implement the style transfer
    # (Additional implementation needed here for extracting features, calculating losses, and updating the target image)

    output = content_image  # Placeholder, replace with the actual output tensor
    save_image(output, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("content_image_path", help="Path to the content image")
    parser.add_argument("style_image_path", help="Path to the style image")
    parser.add_argument("output_path", help="Path to save the output image")
    args = parser.parse_args()
    
    main(args.content_image_path, args.style_image_path, args.output_path)
