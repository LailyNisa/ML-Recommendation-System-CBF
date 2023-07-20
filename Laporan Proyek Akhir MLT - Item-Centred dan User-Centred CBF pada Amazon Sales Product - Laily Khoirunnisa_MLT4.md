##### Laporan Machine Learning Terapan (**Proyek Akhir: Sistem Rekomendasi**)

## **Rekomendasi Produk dengan Metode _Item Centred_ dan _User Centred_ _Content Based Filtering_ pada _Amazon Product Dataset_**

#### Laily Khoirunnisa' - MLT4

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

1. Dataset yang dipakai berjudul _"1K+ Amazon Product's Ratings and Reviews"_. Sumber: [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
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

Pada tahap ini, dilakukan teknik visualisasi data atau exploratory data analysis (EDA).

##### 1. Memetakan jenis atribut, tipe data, dan distribusinya

```sh
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1465 entries, 0 to 1464
Data columns (total 16 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   product_id           1465 non-null   object
 1   product_name         1465 non-null   object
 2   category             1465 non-null   object
 3   discounted_price     1465 non-null   object
 4   actual_price         1465 non-null   object
 5   discount_percentage  1465 non-null   object
 6   rating               1465 non-null   object
 7   rating_count         1463 non-null   object
 8   about_product        1465 non-null   object
 9   user_id              1465 non-null   object
 10  user_name            1465 non-null   object
 11  review_id            1465 non-null   object
 12  review_title         1465 non-null   object
 13  review_content       1465 non-null   object
 14  img_link             1465 non-null   object
 15  product_link         1465 non-null   object
dtypes: object(16)
memory usage: 183.2+ KB
```

Dari output data terlihat bahwa:

- Jumlah data: 1464
- Jumlah atribut: 16.
- Semua data ditampilkan dengan tipe data object.
- Beberapa data numerik perlu diolah lagi agar bisa diketahui korelasinya.

#### 2. Analisis Deskriptif dari Data Numerik

```
    Data actual_price:
    count      1465.000000
    mean       5444.990635
    std       10874.826864
    min          39.000000
    25%         800.000000
    50%        1650.000000
    75%        4295.000000
    max      139900.000000
    Name: actual_price, dtype: float64
```

Dari data di atas, dapat disimpulkan :

- Rata-rata harga produk sekitar 5.445 rupee dari jumlah 1465 produk
- Harga `actual_price` paling rendah 39 rupee, sedangkan paling tinggi 139.900 rupee

```
    Data discounted_price:
    count     1465.000000
    mean      3125.310874
    std       6944.304394
    min         39.000000
    25%        325.000000
    50%        799.000000
    75%       1999.000000
    max      77990.000000
    Name: discounted_price, dtype: float64
```

Dari data di atas, dapat disimpulkan :

- Rata-rata harga setelah diskon dari produk sekitar 3.125 rupee dari jumlah 1465 produk
- Harga `discounted_price` paling rendah 39 rupee, sedangkan paling tinggi 77.990 rupee
- Harga setelah diskon terlihat cukup menarik minat pembeli, karena memiliki rentang dari harga sebelum diskon.

```
    Data discount_percentage:
    count    1465.000000
    mean        0.476915
    std         0.216359
    min         0.000000
    25%         0.320000
    50%         0.500000
    75%         0.630000
    max         0.940000
    Name: discount_percentage, dtype: float64
```

Dari data di atas, dapat disimpulkan :

- Rata-rata harga persen diskon sekitar 48% dari jumlah 1465 produk
- Angka `discount_percentage` paling rendah 0%, sedangkan paling tinggi 94%
- Dapat dilihat, bahwa pemberian diskon menjadi stategi marketing Amazon untuk meningkatkan penjualan produk

```
    Data `rating`:
    count    1465.000000
    mean        4.096519
    std         0.291585
    min         2.000000
    25%         4.000000
    50%         4.100000
    75%         4.300000
    max         5.000000
    Name: rating, dtype: float64
```

    Dari data di atas, dapat disimpulkan :

- Rata-rata rating produk sekitar 4.1 dari jumlah 1465 produk
- Nilai `rating` paling rendah 2, sedangkan paling tinggi 5
- Dapat dilihat, bahwa rata-rata konsumen puas dengan barang yang dibelinya dari Amazon

```
    Data `rating_count`:
    count      1465.000000
    mean      18283.367235
    std       42725.921124
    min           2.000000
    25%        1191.000000
    50%        5179.000000
    75%       17325.000000
    max      426973.000000
Name: rating_count, dtype: float64
```

- Rata-rata jumlah rating produk sekitar 18.283 dari jumlah produk sebesar 1465
- Nilai `rating_count` paling rendah 2, sedangkan paling tinggi 426.973
- Dapat dilihat, bahwa pembeli sangat aktif memberi rating dan feedback dari produk yang dibelinya dari Amazon.

**Korelasi Data Numerik**
![image](https://drive.google.com/uc?export=view&id=1CzOiCplOMQVbuxU2ZH8m74ZSygprZXXQ)
Korelasi fitur pada data Amazon, dapat disimpulkan:

- korelasi tertinggi 0.96 dari `discounted_price` dan `actual_price`.
- korelasi negatif paling besar yakni -0.24 dari `discounted_price` dan `discount_percentage`
- Korelasi terkecil, yaitu 0.011 dari parameter `discount_percentage` dan `rating_count`
- Korelasi negatif terkecil nilai -0.027 dari `rating_count` dan `discounted_price`

#### 2. Analisis Deskriptif dari Data Kategori

Metode yang akan diambil untuk sistem rekomendasi adalah Content-based filtering. Metode ini menghubungkan produk dengan produk lainnya. Dalam menentukan target rekomendasi, ada 3 variabel fitur yang akan diambil karena memiliki banyak kata kunci antara produk satu dengan yg lain, yaitu:

1. Data kategori pada kolom `category`
2. Data review konsumen pada kolom `review`
3. Data nama produk pada kolom `product_name`.
4. Data deskripsi produk pada kolom `about_product`.

**Analisis Kolom Category**
Top 20 Amazon Produk Kategori
![image](https://drive.google.com/uc?export=view&id=1-ZOCgV6tEXUqePHF-wiECoiAqT2MkKSJ)
Dari diagram di atas, disimpulkan bahwa:

- Produk computer dan accessoris-nya sangat menguasai pasar online Amazon dari segi kuantitas
- Jumlah produk terbanyak berikutnya yaitu produk elektronik yang biasa dipakai pengguna secara mobile dan alat elektronik rumah tangga/dapur beserta aksesorisnya.

![image](https://drive.google.com/uc?export=view&id=1pba26vlY_P3e5cm3yLLeBzdu9-pIA79D)

**WordCloud Kolom `product_name`**
![image](https://drive.google.com/uc?export=view&id=1N7YYduWaWiJGu7aRZPFMjMV8YqXew1wg)
Dari WordCloud dapat dilihat bahwa:

- Produk _charger type-C fast charging_ banyak dijual di Amazon
- Nama produk lengkap dengan spesifikasi warna _Black_ dan _White_ banyak dijual di Amazon
- Sebagian besar produk yang dijual adalah alat elektronik. Permintaan konsumen terhadap produk ini cukup besar di pasar online Amazon.

**WordCloud Kolom `review`**
![image](https://drive.google.com/uc?export=view&id=1GmizNPUnaqZ9dTyM7UIA30baWcBFi-9N)
Dari WordCloud `review` dapat dilihat bahwa:

- banyak pembeli mereview produk _cable_ dan produk yang berhubungan dengan _phone_
- Banyak pembeli memberi feedback positif, yaitu _good, best, easy, great_
- banyak pembeli yang puas dengan produk Amazon. Hal ini selaras dengan data nilai rating rata-rata sebesar 4.

**WordCloud Kolom `about_product`**
![image](https://drive.google.com/uc?export=view&id=1JoJhuoVc--lr9ALobgoKuPF1x4Z3ZpmW)
Dari WordCloud `about_product` dapat dilihat bahwa:

- banyak kata keterangan tentang cara menggunakan alat yang dijual
- banyak keterangan yang menggambarkan kelebihan produk, yaitu _easy, perfect, durable, dll_

## Data Preparation

Setelah menganalisa data, dataset Amazon terlihat perlu penyesuaian di banyak bagian sebelum dapat diolah lebih jauh. Penjelasan tahap data preparation terdiri dari pengecekan terhadap missing values, duplikat data, atau data yang tidak konsisten. Pada tahap ini juga dilakukan perubahan tipe data.

### 1. Cek Missing Value

```sh
       Jumlah Missing Value per Variabel:
       product_id             0
       product_name           0
       category               0
       discounted_price       0
       actual_price           0
       discount_percentage    0
       rating                 0
       rating_count           2
       about_product          0
       user_id                0
       user_name              0
       review_id              0
       review_title           0
       review_content         0
       img_link               0
       product_link           0
       dtype: int64
```

Dari tabel di atas, dapat dilihat terdapat 2 missing value pada `rating count`.

```sh
Jumlah Data per Nilai Rating
        4.1    244
        4.3    230
        4.2    228
        4.0    129
        3.9    123
        4.4    123
        3.8     86
        4.5     75
        4       52
        3.7     42
        3.6     35
        3.5     26
        4.6     17
        3.3     16
        3.4     10
        4.7      6
        3.1      4
        5.0      3
        3.0      3
        4.8      3
        3.2      2
        2.8      2
        2.3      1
        |        1
        2        1
        3        1
        2.6      1
        2.9      1
        Name: rating.
```

Dapat dilihat di atas, bahwa ada data tidak konsisten yaitu '|' berjumlah 1. Perlu dilakukan penyesuaian dengan query. Setelah dilihat data produk di link Amazon, nilainya dapat diganti dengan rating '4.0'. Perlu juga dilakukan konversi tipe data menjadi `float`.

Pengecekan data duplikat:

```
    #cek duplikat record
    df_amz.duplicated().sum()
```

Hasil duplikat data bernilai 0. Berarti tidak ada duplikasi data.

Perubahan tipe data objek menjadi numerik perlu dilakukan pada variabel:

1.  `discounted_price` dari object menjadi float. Contoh nilai: ₹3,990. Maka perlu dihilangkan simbol rupee dan tanda ',' sebelum konversi data.
2.  `actual_price` dari object menjadi float. Contoh nilai: ₹3,990. Maka perlu dihilangkan simbol rupee dan tanda ',' sebelum konversi data.
3.  `discount_percentage` dari object menjadi float. Contoh nilai: 90%. Maka perlu dihilangkan simbol '%' dan menormalisasi nilainya, dengan membagi angka diskon dengan 100.
4.  `rating` dari object menjadi float. Contoh nilai: 4,5. Maka perlu dihilangkan tanda ',' sebelum konversi data.
5.  `rating_count` dari object menjadi float. Contoh nilai: 9,324. Maka perlu dihilangkan tanda ',' sebelum konversi data.

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

**Hasil Modelling Item-Centred CBF**

```
Percobaan 1: Rekomendasi untuk Produk B07CRL2GY6
(boAt Rugged V3 Braided Micro USB Cable (Pearl White))

        product_id	    product_name	        category	    score
443     B0789LZTCJ      boAt Rugged v3 Extra..  Computers..	    0.974749
92	    B08HDH26JX	    boAt Deuce USB 300..	Computers..     0.974749
3	    B08HDJ86NZ	    boAt Deuce USB 300..    Computers..	    0.497482
392	    B08HDJ86NZ	    boAt Deuce USB 300..    Computers..	    0.489176
628	    B08HDJ86NZ	    boAt Deuce USB 300..    Computers..	    0.489176
```

> Tabel Rekomendasi 5 Produk untuk Produk B07CRL2GY6 (_boAt Rugged V3 Braided Micro USB Cable_)

Hasil 5 rekomendasi produk terdekat dengan produk B07CRL2GY6, yaitu 5 tipe mikro USB kabel dari merk _boAT_ dengan spesifikasi yang berbeda satu sama lain.

```
Percobaan 2: Rekomendasi untuk Produk B09YL9SN9B (LG TV 80cm 32" HD)

        product_id	product_name	        category	    score
714	    B08DPLCM6T	LG 80cm (32 inches)..	Electronics..	0.626515
168	    B0B9959XF3	Acer 80cm (32 inches).. Electronics..	0.626515
135	    B0B3XY5YT4	LG 108cm (43 inches)..	Electronics..	0.568036
283	    B0B3XXSB1K	LG 139cm (55 inches)..	Electronics..	0.564709
270	    B0B997FBZT	Acer 139cm (55 inches)  Electronics..	0.564709
```

> Tabel Rekomendasi 5 Produk untuk Produk B07CRL2GY6 (_boAt Rugged V3 Braided Micro USB Cable_)

Hasil 5 rekomendasi produk terdekat dengan produk B09YL9SN9B, yaitu 5 tipe televisi dari 3 dari merk yang sama, yaitu LG, sedangkan 2 lainnya berbeda merk: 1 TV dengan spesifikasi sama, 1 TV dengan spesifikasi berbeda.

```
Percobaan 3: Rekomendasi untuk Produk B00J5DYCCA
(Havells Exhaust Fan for Kitchen)

        product_id	product_name	            category	   score
1431	B00KIDSU8S	Havells Ventil Air DX..     Home&Kitchen..	0.469035
1202	B09MT94QLL	Havells.. Ceiling Fan...	Home&Kitchen..	0.435179
1175	B01LYU3BZF	Havells.. Ceiling Fan..	    Home&Kitchen..	0.291960
1287	B095PWLLY6	Crompton.. Ceiling Fan..	Home&Kitchen..	0.285529
1095	B01M0505SJ	Orient.. Ceiling Fan..	    Home&Kitchen..	0.258879

```

> Tabel Rekomendasi 5 Produk untuk Produk B00J5DYCCA (_Havells Exhaust Fan_)

Hasil 5 rekomendasi produk terdekat dengan produk B00J5DYCCA, yaitu 1 item dengan jenis produk yang sama, sedangkan 4 item disajikan dengan jenis yang berbeda, yaitu ceiling fan (bukan exhaust fan). Perbedaan tipe penyajian terjadi karena keterbatasan data.

**Tahapan Proses Memperoleh Rekomendasi User-Centred CBF:**

1. Menghitung cosine similarity
2. Membuat fungsi rekomendasi dengan input parameter `user_id`
   Pada fungsi reomendasi dilakukan tahap berikut:
   - menyimpan index `user_id` sebagai riwayat pembelian produk
   - mengukur cosine similarity dari data index hanya pada data dari `user_id`
   - mencari data `product_name` dan memasukkan di variabel `products` dan memasukannya pada variabel Series ` indices`
   - membuat list urutan dari score produk hasil similarity dan mengambil index dari data yang pernah dibeli user
   - mengambil 5 produk terbesar similarity-nya
   - mengambil data top N (5 data teratas)
3. Mengambil hasil rekomendasi 3 data acak dari `product_id`.

```
Percobaan 1: Rekomendasi untuk User ID (Encoded) 5

 IdEncoded	    recommended product	                             score
1	    5	SanDisk Extreme microSD UHS I Card 128GB for 4...	0.972423
2	    5	SanDisk Ultra® microSDXC™ UHS-I Card, 64GB, 14...	0.348298
3	    5	SanDisk Ultra® microSDXC™ UHS-I Card, 128GB, 1...	0.333741
4	    5	SanDisk Ultra® microSDXC™ UHS-I Card, 256GB, 1...	0.333741
5	    5	SanDisk Ultra® microSDXC™ UHS-I Card, 64GB, 14...	0.333741
```

> Tabel Hasil Rekomendasi UserID 5

UserID 5 pernah membeli memory card di Amazon. Hasil 5 rekomendasi produk terdekat dengan pembelian sebelumnya disajikan 5 produk jenis yang sama, yaitu micro SD card.

```
Percobaan 2: Rekomendasi untuk User ID (Encoded) 11

 IdEncoded	    recommended product	                             score
1	    11	Zebronics Zeb-Companion 107 USB Wireless Keybo...	0.566889
2	    11	Portronics Key2 Combo Multimedia USB Wireless ...	0.565332
3	    11	HP 330 Wireless Black Keyboard and Mouse Set w...	0.548851
4	    11	Zebronics ZEB-KM2100 Multimedia USB Keyboard C...	0.541751
5	    11	Dell USB Wireless Keyboard and Mouse Set- KM33...	0.522301

```

> Tabel Rekomendasi 5 Produk untuk UserID 11

UserID 5 pernah membeli _Zebronics Wired Keyboard and Mouse Combo_ di Amazon. Hasil 5 rekomendasi produk terdekat dengan pembelian sebelumnya disajikan 5 produk yang sama, yaitu keyboard-mouse, dengan 2 item merk yang sama dan 3 item merk lain.

```
Percobaan 3: Rekomendasi untuk User ID (Encoded) 337

 IdEncoded	    recommended product	                             score
1	 337	Bulfyss Plastic Sticky Lint Roller Hair Remove...	0.681677
2	 337	Wolpin 1 Lint Roller with 60 Sheets Remove Clo...	0.643233
3	 337	SHOPTOSHOP Electric Lint Remover, Best Lint Sh...	0.602602
4	 337	House of Quirk Reusable Sticky Picker Cleaner ...	0.556171
5	 337	Portable Lint Remover Pet Fur Remover Clothes ...	0.543243
```

> Tabel Rekomendasi 5 Produk untuk UserID 337

UserID 337 pernah membeli penghilang bulu (_Lint Remover_) di Amazon. Hasil 5 rekomendasi produk terdekat dengan pembelian sebelumnya disajikan 5 jenis produk yang sama, yaitu lint remover dari berbagai merk yang berbeda.

## Evaluation

Tidak seperti klasifikasi dan regresi yang memiliki ukuran kinerja sederhana dan alami, mengevaluasi fungsi peringkat terbukti lebih sulit. Misalkan ada n objek untuk diurutkan. Ukuran evaluasi peringkat harus menginduksi urutan total pada N! kemungkinan hasil peringkat. Telah banyak cara untuk mendefinisikan ukuran peringkat dan ukuran evaluasi telah diusulkan[8][9][10][11][12].

_Normalized Discounted Cumulative Gain_ (NDCG) merupakan salah satu ukuran peringkat yang banyak digunakan dalam aplikasi[13]. NDCG memiliki dua keunggulan dibandingkan dengan banyak tindakan lainnya. Pertama, NDCG memungkinkan setiap dokumen yang diambil memiliki nilai relevansi beragam sementara kebanyakan ukuran peringkat tradisional hanya berupa relevansi biner. Artinya, setiap dokumen dilihat sebagai relevan atau tidak relevan pada ukuran peringkat tradisional, sementara ada tingkat relevansi untuk dokumen di NDCG. Kedua, NDCG melibatkan fungsi diskon di atas peringkat sementara banyak ukuran lainnya secara seragam memberi bobot pada semua posisi. Fitur ini sangat penting untuk mesin pencari karena pengguna lebih memperhatikan dokumen peringkat teratas daripada yang lain[14].

**Apa itu DCG?**

_Discounted cumulative gain_ menimbang setiap skor relevansi berdasarkan posisinya. Rekomendasi di atas mendapatkan bobot yang lebih tinggi sedangkan relevansi rekomendasi di bawah mendapatkan bobot yang lebih rendah.

![image](https://lh4.googleusercontent.com/zA6ypnbEED9XOtheukjRqB5gYAftKvU29KiPUk-AwL9EQXkFZSCyIht0wKwY4WQxbzcv_EbtAXcT86AODXWQ_IShQKhgGcnLBdrwakrKzaxapuGwkb4R1Aw7Vez4P5GbELW3MawRqnM)

Perhatikan bahwa penyebutnya adalah log(i+1) yang memberi bobot lebih pada item yang direkomendasikan di atas.

DCG memiliki satu kekurangan: Skor bergantung pada jumlah item yang direkomendasikan. Jika kami memiliki dua pemberi rekomendasi yang merekomendasikan jumlah item yang berbeda, sulit untuk membandingkan skor DCG. Pemberi rekomendasi dengan lebih banyak item cenderung memiliki skor lebih tinggi[15].

**Apa itu Metrik NDCG?**

_Normalized Discounted Cumulative Gain_ (NDCG) adalah DCG dengan faktor normalisasi dalam penyebut. Penyebutnya adalah skor DCG ideal (IDCG) saat kami merekomendasikan item yang paling relevan terlebih dahulu[15].

![image](https://lh5.googleusercontent.com/Vw1V56DgbbYjdezCVkoon5LGj9xsRJsWum0TECdoN1Vcj4UBkR66cBfVvA_Yy842ws27oy4UZq_r0xsLy7DEB62wfIYTLf_YKW9qd5B8nZCPdMrDNiIjmaBv0NwscSfVoJBSYJwUgxQ)

Berikut hasil evaluasi sistem rekomendasi yang telah dibuat dengan NDCG.

```
    	   Evaluasi Item Centred Content-Based Filtering dengan NDCG
    	        R1	R2	R3	R4	R5
    PRED 1	    3	3	1	1	1
    TRUE 1	    3	2	2	2	2
    PRED 2	    2	2	1	1	1
    TRUE 2	    3	2	1	1	0
    PRED 3  	1	1	1	1	1
    TRUE 3  	2	1	1	0	0

                DCG	        ICG	        Hasil NDCG
Percobaan 1     6.21	    6.90	    0.90
Percobaan 2     4.58	    5.19	    0.88
Percobaan 3     2.95	    3.13	    0.94
```

```
     Evaluasi User Centred Content-Based Filtering dengan NDCG
    	        R1	R2	R3	R4	R5
    PRED 1	    3	1	1	1	1
    TRUE 1	    2	2	2	2	2
    PRED 2	    2	2	2	2	2
    TRUE 2  	3	2	1	3	1
    PRED 3  	2	2	2	2	2
    TRUE 3  	3	2	2	3	2


                 DCG	    ICG         Hasil NDCG
Percobaan 1     4.95	    5.90    	0.84
Percobaan 2     5.90	    6.44    	0.92
Percobaan 3     5.90	    7.33    	0.80

```

## Kesimpulan

1. Rata-rata hasil evaluasi peringkat rekomendasi dengan NDCG dari Item Centred CBF sebesar 91%.
2. Rata-rata hasil evaluasi peringkat rekomendasi dengan NDCG dari User Centred CBF sebesar 85.3%.
3. Hasil percobaan bisa berbeda-beda tergantung dari seberapa banyak fitur data input dan keragaman data pada dataset.
4. Semakin banyak data dengan kemiripan satu-sama lain, maka semakin baik hasil yang diperoleh NDCG. Hal ini sesuai dasar teori yang telah dijelaskan sebelumnya.

## Daftar Pustaka

[1] J. Karkavelraja. 2023. [_Amazon Sales Dataset_.](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)

[2] Luthfi,2023. [_Apa itu Amazon?_](https://www.ahlibelanja.com/apa-itu-amazon/)

[3] Zuzanna Deutschman.2023. [_Recommender Systems: Machine Learning Metrics and Business Metrics_](https://neptune.ai/blog/recommender-systems-metrics)

[4] Sharma, Himanshu.2022. [_Practical Tips to Develop User Engagement for E-commerce_](https://www.optimizesmart.com/practical-tips-develop-user-engagement-e-commerce-site/)

[5] Qaiser, S., & Ali, R. (2018). _Text Mining: Use of TF-IDF to Examine the Relevance of Words to Documents._ International Journal of Computer Applications, 25-29.

[6] Samuel, R., Natan, R., Fitria, & Syafiqoh, U. (2018). _Penerapan Cosine Similarity dan KNearest Neighbor (K-NN) pada Klasifikasi dan Pencarian Buku_. Journal of Big Data Analytic and Artificial Intelligence, 9-14.

[7] Rymmai, R. G., & JS, S. (2017). _Book Recommendation Using Cosine Similarity_. International Journal of Advanced Research in Computer Science, 276-281.

[8] O. Chapelle, D. Metlzer, Y. Zhang, and P. Grinspan. _Expected reciprocal rank for graded relevance_. In Proceedings of the 18th ACM conference on Information and knowledge management, pages 621–630. ACM, 2009.

[9] A. Turpin and F. Scholer. _User performance versus precision measures for simple search tasks._ In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval, pages 11–18. ACM, 2006.

[10] R. Baeza-Yates and B. Ribeiro-Neto. _Modern Information Retrieval_, volume 82. AddisonWesley New York, 1999.

[11] S. Agarwal, T. Graepel, R. Herbrich, S. Har-Peled, and D. Roth. _Generalization bounds for the area under an ROC curve_. 2004.

[12] C. Rudin. _The p-norm push: A simple convex ranking algorithm that concentrates at the top of the list._ The Journal of Machine Learning Research, 10:2233–2271, 2009.

[13] K. J¨arvelin and J. Kek¨al¨ainen. \_Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems (TOIS), 20(4):422–446, 2002.

[14] Wang Yining, Wang Liwei.2013.[_A Theoretical Analysis of NDCG Ranking Measures_](http://proceedings.mlr.press/v30/Wang13.pdf)

[15] MLNerds.2021. [_NDCG Evaluation Metric for Recommender Systems_](https://machinelearninginterview.com/topics/machine-learning/ndcg-evaluation-metric-for-recommender-systems/)
