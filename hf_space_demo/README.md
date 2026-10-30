---
title: SIBI Recognition VideoMAE Demo
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# SIBI Recognition VideoMAE Space

Demo ini menggunakan Hugging Face Spaces dengan Gradio untuk menjalankan klasifikasi gerakan Bahasa Isyarat Indonesia (SIBI).

## Cara deploy ke Hugging Face Space
1. Buat Space baru di Hugging Face:
   - Pilih **Gradio** sebagai SDK
   - Hardware: CPU basic (atau GPU bila tersedia)
   - Nama repo: `SIBI_Recognition_VideoMAE-demo`

2. Pastikan file-file berikut ada di root repo Space:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - Folder `VideoMAE/`

3. Model Checkpoint:
   - Checkpoint model otomatis diunduh dari Hugging Face Hub: `Luckystg/SIBI_Recognition_VideoMAE` (`checkpoint-best.pth`).

## Cara Menggunakan
- **Upload File**: Unggah video (.mp4, .webm, .avi, .mov) berisi satu gerakan isyarat tangan.
- **Rekam Webcam**: Gunakan kamera langsung melalui browser untuk merekam gerakan isyarat.
- Klik Submit / tunggu prediksi selesai untuk melihat Top-5 kelas probabilitas.

## Kelas SIBI yang Dikenali (10 Kata)
1. Aku
2. Bapak
3. Bawa
4. Guru
5. Ibu
6. Kamu
7. Mau
8. Pulang
9. Sekolah
10. Suka

## Catatan Teknis
- Arsitektur: `vit_base_patch16_224` (VideoMAE fine-tuned)
- Input: 16 frame temporal (ImageNet normalized)
- Multi-engine video loader (PyAV + OpenCV + Decord) yang kompatibel dengan format WebM webcam browser dan MP4 upload.
