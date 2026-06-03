# SIBI Recognition VideoMAE Space

Demo ini menggunakan Hugging Face Spaces dengan Gradio untuk menjalankan inferensi video.

## Cara deploy
1. Buat Space baru di Hugging Face:
   - Pilih `Gradio` sebagai framework
   - Pilih `CPU` atau `GPU` jika ingin akurasi lebih cepat
   - Nama repo contoh: `SIBI_Recognition_VideoMAE-demo`

2. Upload semua file dari folder `hf_space_demo` ke repo Space:
   - `app.py`
   - `requirements.txt`
   - folder `VideoMAE/` (copy seluruh kode package VideoMAE)

3. Pastikan model checkpoint tersedia:
   - Jika repo `Luckystg/SIBI_Recognition_VideoMAE` publik, app akan otomatis download dari repo model.
   - Jika repo model private, tambahkan secret `HF_TOKEN` di Space dan set environment variable `HUGGINGFACE_HUB_TOKEN`.

## Cara menggunakan
- Buka Space Anda di browser
- Upload file video `.mp4`
- Tunggu prediksi muncul sebagai Top-5 label

## Catatan teknis
- Model yang digunakan: `vit_base_patch16_224`
- Input video diproses menjadi 16 frame dengan sampling dense
- Output: probabilitas kelas dan prediksi utama

## Struktur file Space
- `app.py` : aplikasi Gradio
- `requirements.txt` : dependencies
- `VideoMAE/` : paket model dan utilitas VideoMAE
