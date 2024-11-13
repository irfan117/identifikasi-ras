import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Memuat model
model = load_model('rasvvv2.h5')

# nama ras
race_dict = {0: 'African', 1: 'American', 2: 'Asia Tenggara', 3: 'Asia Timur', 4: 'Eropa'}

# Fungsi untuk memproses frame dan melakukan prediksi
def preprocess_and_predict(face_img, model):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    predictions = model.predict(face_img)
    predicted_race_index = np.argmax(predictions, axis=1)[0]
    return race_dict[predicted_race_index]

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Muat model deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame, keluar...")
        break

    # Konversi ke skala abu-abu untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Ekstrak wajah dan lakukan prediksi
        face_img = frame[y:y+h, x:x+w]
        predicted_race = preprocess_and_predict(face_img, model)

        # Buat overlay untuk grafis
        overlay = frame.copy()
        sci_fi_color = (173, 216, 230)  # Warna biru muda (BGR)

        # Menambahkan persegi panjang dengan sudut-sudut tidak menyatu
        thickness = 2
        cv2.rectangle(overlay, (x, y), (x + w//4, y + thickness), sci_fi_color, thickness)  # Sudut kiri atas
        cv2.rectangle(overlay, (x, y), (x + thickness, y + h//4), sci_fi_color, thickness)
        cv2.rectangle(overlay, (x + w, y), (x + w - w//4, y + thickness), sci_fi_color, thickness)  # Sudut kanan atas
        cv2.rectangle(overlay, (x + w - thickness, y), (x + w, y + h//4), sci_fi_color, thickness)
        cv2.rectangle(overlay, (x, y + h), (x + w//4, y + h - thickness), sci_fi_color, thickness)  # Sudut kiri bawah
        cv2.rectangle(overlay, (x, y + h - h//4), (x + thickness, y + h), sci_fi_color, thickness)
        cv2.rectangle(overlay, (x + w, y + h), (x + w - w//4, y + h - thickness), sci_fi_color, thickness)  # Sudut kanan bawah
        cv2.rectangle(overlay, (x + w - thickness, y + h), (x + w, y + h - h//4), sci_fi_color, thickness)

        # Menambahkan teks identifikasi di bawah wajah
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = sci_fi_color
        thickness = 2
        text = f'Ras Teridentifikasi: {predicted_race}'
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + h + text_size[1] + 20

        # Menambahkan latar belakang kotak untuk teks
        cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), cv2.FILLED)

        # Menambahkan teks ke frame
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Menambahkan efek transparan
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Menampilkan frame dengan overlay
    cv2.imshow('Real-time Race Identification', frame)

    # menghentikan identifikasi press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan objek
