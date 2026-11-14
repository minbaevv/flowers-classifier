import base64
from fastapi import FastAPI, HTTPException,UploadFile,File
import io
import uvicorn
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import json

with open("classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

app = FastAPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device)
model.fc = torch.nn.Linear(512, len(classes))
model.load_state_dict(torch.load("flowers_cnn.pth", map_location=device))
model.to(device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail='Файл кошулган жок')

        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = torch.argmax(y_pred).item()

        return {"result": classes[pred]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



