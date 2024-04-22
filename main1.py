import argparse
import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

def main(content_image_path, style_image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = load_image(content_image_path).to(device)
    style_image = load_image(style_image_path, scale=0.5).to(device)

    model = models.vgg19(pretrained=True).features
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.MaxPool2d):
            model[int(name)] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    model.to(device).eval()

    # Obtain initial content and style features
    content_features = get_features(content_image, model)
    style_features = get_features(style_image, model)
    target = content_image.clone().requires_grad_(True)
    
    # Style weights
    style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
    
    # Optimizer
    optimizer = optim.Adam([target], lr=0.003)

    for i in tqdm(range(300)):
        optimizer.zero_grad()  # Clear existing gradients before forward pass
        target_features = get_features(target, model)
        content_loss = torch.mean((target_features['conv4_1'] - content_features['conv4_1'])**2)

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_features[layer])
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (target_feature.shape[1] * target_feature.shape[2])
        
        total_loss = content_loss + style_loss
        total_loss.backward(retain_graph=True)  # Retain the computation graph after backward
        optimizer.step()  # Apply gradient descent to update the target image

    
    save_image(target, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer with PyTorch")
    parser.add_argument("content_image_path", type=str, help="Path to the content image")
    parser.add_argument("style_image_path", type=str, help="Path to the style image")
    parser.add_argument("output_path", type=str, help="Path to save the output image")
    args = parser.parse_args()
    
    main(args.content_image_path, args.style_image_path, args.output_path)
