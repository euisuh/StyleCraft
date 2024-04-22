from PIL import Image
from torchvision import transforms

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