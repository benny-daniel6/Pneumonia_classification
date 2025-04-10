import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets,models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_data=datasets.ImageFolder(root='data/train',transform=train_transform)
test_data=datasets.ImageFolder(root='data/test',transform=test_transform)
train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(test_data,batch_size=32)
print("Train Class Count : ",[len(train_data.imgs[i]) for i in range(len(train_data.classes))])
model=models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad=False
num_fltrs=model.fc.in_features
model.fc=nn.Sequential(
    nn.Linear(num_fltrs,256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256,1),
    nn.Sigmoid()
)
model=model.to(device)
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
def train_model(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        for inputs,labels in train_loader:
            inputs,labels=inputs.to(device),labels.to(device).float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()    
            running_loss+=loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader) : .4f}")
train_model(epochs=10)

def evaluate_model():
    model.eval()
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            predicted=(outputs>0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true,y_pred,target_names=["NORMAL","PNEUMONIA"]))
    cm=confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=["NORMAL","PNEUMONIA"],yticklabels=["NORMAL","PNEUMONIA"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("confusion_matrix.png")
evaluate_model()
torch.save(model.state_dict(),"pneumonia_resnet18.pth")
# !pip install grad-cam
from gradcam import GradCAM

def visualize_gradcam(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model=model, target_layer=model.layer4[-1])
    mask, _ = gradcam(img_tensor)
    
    # Overlay heatmap on image
    heatmap = (mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + np.array(img) * 0.6
    
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

visualize_gradcam("chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg")