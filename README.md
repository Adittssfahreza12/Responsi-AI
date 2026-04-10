# Animal Classifier Web App

Aplikasi web Flask untuk prediksi gambar hewan berdasarkan model CNN dari notebook `animal_classifier_Aditya Fahreza (1).ipynb`.

## Fitur
- Upload gambar hewan
- Prediksi kelas hewan
- Menampilkan confidence dan probabilitas tiap kelas
- UI modern dan responsif

## Struktur Model
Sama seperti notebook:
- Conv2d 3x3 -> MaxPool -> ReLU
- Conv2d 3x3 -> MaxPool -> ReLU
- Conv2d 3x3 -> MaxPool -> ReLU
- Flatten
- Linear 128
- Output klasifikasi

## Cara pakai
1. Latih model di notebook.
2. Simpan checkpoint ke file `animal_classifier.pth` dengan format berikut:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": label_encoder.classes_.tolist()
}, "animal_classifier.pth")
```

3. Letakkan `animal_classifier.pth` di folder project ini.
4. Install dependency:
```bash
pip install flask torch torchvision pillow
```

5. Jalankan:
```bash
python app.py
```

6. Buka browser ke:
```bash
http://127.0.0.1:5000
```

## Catatan
- Preprocessing inference memakai `Resize((128,128))` dan `ToTensor()`, mengikuti ukuran input notebook.
- Saat inference tidak dipakai augmentasi acak agar hasil lebih stabil.
