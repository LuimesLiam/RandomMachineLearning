import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
from PIL import Image
from Models import SimpleCNN, ImprovedCNN



batch_size = 32
image_size = (224, 224)  # Resize all images to 224x224 pixels
num_epochs = 2
learning_rate = 0.01
validation_split = 0.2
num_classes = len(os.listdir('FlowerClassifcation/flower_photos'))  # Count the number of classes in your dataset

transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='FlowerClassifcation/flower_photos', transform=transform)
dataset_size = len(dataset)
val_size = int(dataset_size * validation_split)
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset,batch_size =batch_size, shuffle=True)
val_loader = DataLoader (val_dataset, batch_size = batch_size, shuffle=False)

#init the model 

model = SimpleCNN(num_classes) #ImprovedCNN(num_classes)

criterion = nn.CrossEntropyLoss() 

optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() 

            output = model(images) 
            loss =criterion(output,labels)
            
            loss.backward()
            optimizer.step()


            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}"
              f"\nPredicted:{predicted}:\n{labels}")

    return train_losses, val_losses

train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

torch.save(model.state_dict(), 'flower_classifier.pth')

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


#_________________________________________________________________
inference = False

if (inference):
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
    
    return class_names[predicted.item()]

if (inference):
    image_path = 'FlowerClassifcation/flower_photos/tulips/10791227_7168491604.jpg'
    predicted_class = predict_image(image_path, model)
    print(f'The predicted class for the image is: {predicted_class}')







