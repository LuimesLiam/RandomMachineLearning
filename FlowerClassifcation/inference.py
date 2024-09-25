import torch
from torchvision import transforms
from PIL import Image
import os
from Models import SimpleCNN, ImprovedCNN

num_classes = len(os.listdir('FlowerClassifcation/flower_photos')) 
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('flower_classifier.pth'))
model.eval() 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1) 
    
    class_names = os.listdir('FlowerClassifcation/flower_photos')
    
    print("\n", predicted, predicted.item(), class_names)
    return class_names[predicted.item()-1]

image_path = 'FlowerClassifcation/flower_photos/sunflowers/58636535_bc53ef0a21_m.jpg'  
predicted_class = predict_image(image_path, model)
print(f'The predicted class for the image is: {predicted_class}')
