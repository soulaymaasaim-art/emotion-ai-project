from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = 'ton_cle_secrete_ici'  # Nécessaire pour la session

# -------------------
# Configuration Login
# -------------------
users = {
    "prof": "emotion123",
    "student": "motdepasse"
}

# -------------------
# Dossier pour les uploads
# -------------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------
# Chargement du modèle
# -------------------
model = load_model('emotion_model.h5')
emotion_dict = {0: 'happy', 1: 'sad'}

# -------------------
# Routes
# -------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].lower()
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Nom d’utilisateur ou mot de passe incorrect'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    result = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Prétraitement image
            img = image.load_img(file_path, target_size=(48,48), color_mode="grayscale")
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)/255.0

            # Prédiction
            prediction = model.predict(img_array)
            result = emotion_dict[int(np.round(prediction[0][0]))]

    return render_template('index.html', result=result, filename=filename)

# -------------------
# Lancer l'app
# -------------------
if __name__ == '__main__':
    app.run(debug=True)













