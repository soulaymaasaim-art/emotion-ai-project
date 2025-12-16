from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image, ImageOps

app = Flask(__name__)

# Dossier pour les images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle
model = load_model('emotion_model.h5')

# Récupérer l'ordre des classes automatiquement
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
train_dir = 'dataset/train'
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)
class_indices = train_generator.class_indices
print("Indices des classes :", class_indices)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    predicted_emotion = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return "Aucune image sélectionnée !"

        filename = file.filename
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # Prétraitement amélioré pour images réelles
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)           # convertir en grayscale
        img = img.resize((48,48))               # redimensionner
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normaliser

        # Prédiction
        prediction = model.predict(img_array)[0][0]

        # Interprétation selon l'ordre des classes
        if class_indices['happy'] == 0:
            predicted_emotion = 'happy' if prediction < 0.5 else 'sad'
            confidence = (1 - prediction) * 100 if predicted_emotion=='happy' else prediction*100
        else:
            predicted_emotion = 'happy' if prediction > 0.5 else 'sad'
            confidence = prediction*100 if predicted_emotion=='happy' else (1 - prediction)*100

        result = f"L'émotion prédite : {predicted_emotion.upper()} ({confidence:.1f}% de confiance)"

    return render_template('index.html', result=result, filename=filename, predicted_emotion=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)








