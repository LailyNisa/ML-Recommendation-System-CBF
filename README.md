# ML-Recommendation-System-CBF

## Project Overview

Amazon adalah Perusahaan Multi-Nasional Teknologi Amerika yang kepentingan bisnisnya mencakup perdagangan berbasis elektronik (_e-commerce_). Amazon menjadi tempat user membeli dan menyimpan inventaris, dan menangani semuanya mulai dari pengiriman, penetapan harga hingga layanan pelanggan dan pengembalian [1]. Produk yang dijual dapat diberi rating/penilaian dan review/ulasan tentang suatu produk. Semua data tersebut dapat dilihat secara realtime di website Amazon.

Sebagai sebuah retail online terbaik, Amazon telah berubah dari sebuah toko menjadi pasar. Artinya perusahanan milik Bezos menyediakan tempat untuk penjual menampilkan barang dagangan, sedangkan pembeli bebas untuk langsung mengecek dan membelinya. Siapapun bisa ikut mengakses website raksasa _e-commerce_ ini asalkan sudah mempunyai akun. Hal itu membuat berbagai negara memilihnya sebagai marketplace favorit[2]. 

Sistem pemberi rekomendasi bertujuan untuk menyarankan konten atau produk yang relevan kepada pengguna yang mungkin disukai atau dibeli oleh mereka. Ini membantu untuk menemukan item yang dicari pengguna. Dan pengguna bahkan tidak menyadarinya hingga rekomendasi ditampilkan. Strategi yang berbeda harus diterapkan untuk klien yang berbeda dan ditentukan oleh data yang tersedia. Karena sistem rekomendasi harus menjadi pendekatan berbasis data, sistem rekomendasi didorong oleh algoritme machine learning[3].

## Business Understanding
Kekuatan rekomendasi produk untuk mendapatkan hasil maksimal sistem rekomendasi dan meningkatkan pengalaman pengguna, harus berdasarkan pemahaman dan pendalaman hubungan antara:
1. Pengguna dan produk
2. Produk dan produk
3. Pengguna dan pengguna.

Menjaga hubungan ini dalam pikiran saat merancang sistem rekomendasi, akan menghasilkan pengalaman yang menyenangkan bagi pengguna dan akibatnya meningkatkan keterlibatan mereka dengan produk semacam itu. Bayangkan YouTube tanpa rekomendasi video yang Anda sukai. Sebagian besar dari kita menghabiskan banyak waktu di sana hanya karena rekomendasinya sangat akurat![3]

Keterlibatan pengguna akan berefek pada kunjungan berulang yang akan membuat kecenderungan terhadap produk/brand/merek. Kecenderungan terhadap merek akan berefek pada loyalitas merek yang membawa dampak besar pada kenaikan nilai penjualan, prospek, dan jumlah pelanggan.[4]

### Problem Statements
Pernyataan masalah:     
- Bagaimana membuat sistem rekomendasi produk Amazon berdasarkan item produk dengan metode Content Based Filtering?
- Bagaimana membuat sistem rekomendasi produk Amazon berdasarkan riwayat pembelian user dengan metode Content Based Filtering?
- Bagaimana mengukur akurasi dari sistem rekomendasi yang telah dibuat?

### Goals
Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Membuat rekomendasi produk berdasarkan pemilihan nama produk dengan metode _Content Based Filtering._
- Membuat rekomendasi produk berdasarkan riwayat pembelian dari user dengan metode _Content Based Filtering._
- Mengetahui sejauh mana akurasi dari sistem rekomendasi yang telah dibuat.

    ### Solution statements
    - Menggunakan tf-idf dan cosine similarity untuk mendapatkan top 5 dari kemiripan produk yang diukur dengan data nama produk, kategori, review dan tentang produk.
    - Menggunakan tf-idf dan cosine similarity untuk mendapatkan top 5 kemiripan produk sesuai riwayat pembelian user_id tertentu, yang diukur dengan data nama produk, kategori, review dan tentang produk.
    - Menghitung evaluasi rekomendasi dengan NDCG  

## Data Understanding
1. Dataset yang dipakai berjudul _"1K+ Amazon Product's Ratings and Reviews"_. Sumber:  [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
2. Variabel dataset :
    - product_id - ID dari produk yang dijual (object)
    - product_name - Nama produk yang dijual (object)
    - category - produk kategori yang dipisah dengan karakter '|' (object)
    - discounted_price - harga setelah diskon (object)
    - actual_price - harga produk yang ditampilkan (object)
    - discount_percentage - Percentasi diskon produk (object)
    - rating - Rating/penilaian produk (object)
    - rating_count - Jumlah user yang menilai produk (object)
    - about_product - Deskripsi produk (object)
    - user_id - ID user yang menulis review produk, nilai bisa lebih dari 1 user dengan dipisah tanda ',' (object)
    - user_name - Nama user yang me-review produk, nilai bisa lebih dari 1 nama dengan dipisah tanda ',' (object)
    - review_id - ID review, nilai bisa lebih dari 1 review ID dengan dipisah tanda ',' (object)
    - review_title - Judul review pendek, merupakan kumpulan review dari user dengan rating sama yang dipisah tanda ',' (object)
    - review_content - Detail isi review (object) 
    - img_link - Image Link dari produk
    - product_link - Official Website Link dari produk
  
## Data Preparation
### 1. Cek Missing Value
### 2. Data cleaning
Step yang akan dilakukan, yaitu :
1. Mengisi data kosong, 
2. menghilangkan spasi kosong pada data `product_id`, 
3. melakukan penyesuaian data pada inkonsisten data.

### 3. Text-Preprocessing
1. Memisah kata kunci pada kolom `category` yang sebelumnya ditulis tanpa spasi dan kata hanya dibedakan dengan huruf besar dan simbol. Hal ini perlu dilakukan agar tidak terjadi kesalahan pengambilan kata kunci.
2. Membuat fungsi preprocess siap panggil, yang akan memproses input, sebagai berikut :
    - mengkonversi huruf menjadi huruf kecil semua
    - menghilangkan angka
    - memanggil library RegexpTokenizer yang akan memecah input menjadi kata-per kata
    - mengembalikan value berupa penggabungan hasil token 
3. Mengaplikasikan fungsi preprocess pada data `product_name`, `category`, `review`, dan `about_data`.
4. Menggabungkan hasil poin ke-3 menjadi 1 variabel.
5. Mengubah fitur text menjadi fitur numerik dengan TF-IDF. TF-IDF merupakan implementasi dari statistik numerik yang menunjukkan relevansi kata kunci dengan beberapa dokumen tertentu, dengan menyediakan kata kunci yang tersedia, beberapa dokumen tertentu dapat diidentifikasi atau dikategorikan sesuai dengan relevansinya [5].
6. Untuk user centred CBF, dilakukan encoding `user_id` dengan LabelEncoder karena data `user_id` cukup rumit dan panjang.

## Modeling dan Results
Sistem rekomendasi produk Amazon dengan metode _Item centred_ dan _User Centred_ _Content-based filtering_ akan menyajikan top-5 rekomendasi produk sebagai output.
 
**Content-Based Filtering**
Content-based filtering suatu sistem reomendasi akan menyarankan item produk yang mirip dengan suatu produk yang disukai atau pun yang telah dibeli (strategi kontekstual). Misalnya, jika user A ingin melihat film horor, maka film horor lain akan ditampilkan untuknya. Teknik ini dapat item-centred ataupun user-centred[3].
1. Item Centred Content-based filtering 
    _Item-centred content-based filtering_ dari sistem rekomendasi akan mereomendasikan item baru berdasarkan kesamaan item saja (feedbac implisit)[3].
    ![image](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Recommender-Systems-ML-Metrics-vs-Business-Metrics_29.png?resize=273%2C172&ssl=1)
2. User Centred Content-based filtering
   Pada kasus _user-centered content-based filtering_, informasi user dikumpulkan, misalnya melalui kuisioner. Dari situlah pengetahuan dibawa untuk merekomendasikan item dengan fitur yang serupa yang ia sukai[3].
    ![image](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Recommender-Systems-ML-Metrics-vs-Business-Metrics_28.png?resize=319%2C219&ssl=1)

**Kelebihan dan Kekurangan CBF:**
Contoh : Jika pengguna menyukai komedi, komedi lain direkomendasikan
Keuntungan : Tidak memerlukan data apa pun tentang pengguna lain, dapat merekomendasikan item khusus
Kekurangan : Membutuhkan banyak pengetahuan domain, membuat rekomendasi hanya berdasarkan minat pengguna yang ada[3].

Setelah memproses text dengan TF-IDF, sistem rekomendasi CBF item centred akan dihitung dengan algoritma kemiripan _Cosine SImilarity_. 

**Cosine Similarity**
Metode _Cosine Similarity_ adalah mengukur kemiripan antara dua dokumen atau teks. Pada _cosine similarity_ dokumen atau teks dianggap sebagai vektor[6]. Dalam pengertian lain, cosine similarity antara dua vektor (atau dua dokumen pada vector space) merupakan pengukuran yang menghitung antara sudut kosinus di antara keduanya. Metrik ini adalah pengukuran orientasi dan bukan besarannya, dapat dilihat sebagai perbandingan antar dokumen pada ruang yang dinormalisasi karena tidak hanya besaran setiap hitungan kata (TF-IDF) yang dipertimbangkan dari setiap dokumen, tetapi sudut antar dokumen-dokumen[7]. 

**Tahapan Proses Memperoleh Rekomendasi Item-Centred CBF:**
1. Menghitung cosine similarity
2. Membuat fungsi rekomendasi dengan input parameter `product_id`
    Pada fungsi reomendasi dilakukan tahap berikut:
    - menyimpan index `product_id`
    - mengukur cosine similarity dari data index.
    - membuat list dari urutan score produk hasil similarity
    - mengurutkan score
    - mengambil data top N (5 data teratas) 
3. Mengambil hasil rekomendasi 3 data acak dari `product_id`.
