# Submission 1: Cross Site Scripting (XSS) Detection
Nama: Harry Mardika

Username dicoding: hkacode

| | Deskripsi |
| ----------- | ----------- |
| **Dataset** | [Cross site scripting XSS dataset for Deep learning](https://www.kaggle.com/datasets/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning) |
| **Masalah** | Cross-site scripting (XSS) adalah jenis serangan siber yang memungkinkan penyerang menyisipkan skrip berbahaya ke dalam halaman web yang dilihat oleh pengguna lain. Serangan XSS dapat menyebabkan pencurian informasi, pengambilalihan akun, dan kerentanan keamanan lainnya. Masalah ini penting untuk diatasi karena dapat membahayakan data dan privasi pengguna. Dalam proyek ini, kita akan mendeteksi serangan XSS secara otomatis menggunakan model pembelajaran mendalam (deep learning). |
| **Solusi Machine Learning** | Solusi yang diusulkan adalah mengembangkan model pembelajaran mendalam untuk mengklasifikasikan input teks sebagai aman atau berbahaya. Model akan dilatih menggunakan dataset XSS untuk mengenali pola-pola yang umum digunakan dalam serangan XSS. Dengan menggunakan deep learning, model diharapkan dapat mendeteksi serangan XSS secara efektif dan efisien. |
| **Metode Pengolahan** | Metode pengolahan data yang digunakan dalam proyek ini meliputi: <br> 1. **Lowercasing**: Mengubah semua teks menjadi huruf kecil untuk memastikan konsistensi dalam analisis. <br> 2. **Cleaning**: Menghapus tanda petik satu (') dan petik dua (") dari input untuk mengurangi kompleksitas karakter spesial yang bisa menyebabkan kesalahan parsing. <br> 3. **Tokenization**: Memecah teks menjadi token-token yang lebih kecil yang dapat dianalisis secara individual. <br> 4. **Vectorization**: Mengubah token-token ini menjadi representasi numerik yang dapat digunakan sebagai input untuk model pembelajaran mendalam. |
| **Arsitektur Model** | Arsitektur model yang digunakan terdiri dari beberapa lapisan, yaitu: <br> 1. **Input Layer**: Menerima input teks dalam bentuk string. <br> 2. **Text Vectorization Layer**: Mengubah teks menjadi token numerik dengan TextVectorization layer dari Keras. <br> 3. **Embedding Layer**: Mengubah token numerik menjadi vektor dimensi yang lebih tinggi untuk menangkap makna kontekstual dari token. <br> 4. **Convolutional Layer**: Menggunakan lapisan Conv1D untuk menangkap fitur spasial dari teks, membantu dalam mendeteksi pola-pola umum dalam serangan XSS. <br> 5. **Global Max Pooling Layer**: Mengambil nilai maksimum dari fitur yang dideteksi untuk setiap filter, mengurangi dimensi data dan fokus pada fitur yang paling menonjol. <br> 6. **Dense Layer**: Menggunakan lapisan fully connected untuk menggabungkan fitur-fitur yang diekstraksi dari lapisan sebelumnya. <br> 7. **Output Layer**: Menggunakan lapisan Dense dengan aktivasi sigmoid untuk menghasilkan probabilitas antara 0 (aman) dan 1 (berbahaya). |
| **Metrik Evaluasi** | Metrik yang digunakan untuk mengevaluasi performa model adalah **binary crossentropy** dan **accuracy**. <br> 1. **Binary Crossentropy**: Mengukur kerugian (loss) antara label sebenarnya dan prediksi model. Rumusnya adalah: <br> \[ \text{Binary Crossentropy} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)] \] di mana \( y_i \) adalah label sebenarnya, \( p_i \) adalah probabilitas prediksi, dan \( N \) adalah jumlah sampel. <br> 2. **Accuracy**: Mengukur persentase prediksi yang benar dari total prediksi yang dibuat. Rumusnya adalah: <br> \[ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \] Metrik ini memberikan gambaran umum tentang seberapa baik model dalam mengklasifikasikan input sebagai aman atau berbahaya. |
| **Performa Model** | Model yang dibuat mencapai performa yang baik pada data uji dengan hasil sebagai berikut: <br> - **Loss**: 0.0754 <br> - **Accuracy**: 0.9750 <br> - **Validation Loss**: 0.0330 <br> - **Validation Accuracy**: 0.9907 <br> Performa ini menunjukkan bahwa model dapat mengenali pola-pola umum dalam serangan XSS dengan akurasi yang sangat tinggi. Validation loss yang lebih rendah dibandingkan training loss mengindikasikan bahwa model tidak overfitting dan dapat menggeneralisasi dengan baik pada data baru. Akurasi yang tinggi baik pada training maupun validation set menunjukkan bahwa model dapat melakukan klasifikasi dengan tepat dan konsisten. |
