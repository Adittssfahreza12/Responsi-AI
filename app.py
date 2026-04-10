from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import urllib.request

app = Flask(__name__)

# ====== LOAD MODEL ======
model = models.resnet18(weights="DEFAULT")

# batasi jadi 3 kelas (cat, dog, wild)
model.fc = nn.Linear(model.fc.in_features, 3)

model.eval()

# ====== LABEL KELAS (SESUAIKAN DENGAN DATASETMU) ======
class_names = ['cat', 'dog', 'wild']  # ganti sesuai dataset kamu

# ====== TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====== HTML + CSS (INLINE) ======
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Animal Classifier</title>
    <style>
        body {
            font-family: Arial;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            text-align: center;
            padding: 40px;
        }
        .container {
            background: white;
            color: black;
            padding: 30px;
            border-radius: 15px;
            width: 400px;
            margin: auto;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        input {
            margin: 15px;
        }
        button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        img {
            margin-top: 20px;
            max-width: 100%;
            border-radius: 10px;
        }
        h2 {
            margin-top: 20px;
        }
    </style>
</head>
<body>

    <h1>🐾 Animal Classifier AI</h1>

    <div class="container">
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br>
            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <h2>Hasil: {{ prediction }}</h2>
            <h3>Akurasi: {{ confidence }}%</h3>
            <img src="data:image/png;base64,{{ image }}">
        {% endif %}
    </div>

</body>
</html>
"""

# ====== ROUTE ======
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        import base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return render_template_string(
            HTML,
            prediction=class_names[predicted.item()],
            confidence=round(confidence.item()*100, 2),
            image=img_base64
        )

    return render_template_string(HTML, prediction=None)

# ====== RUN ======
if __name__ == '__main__':
    app.run(debug=True)