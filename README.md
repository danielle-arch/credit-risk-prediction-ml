# Credit Risk Prediction Model 🏦
*A Machine Learning Portfolio for Data Science Internship*

## 📌 Project Overview
Proyek ini membangun model klasifikasi Machine Learning (XGBoost) untuk memprediksi probabilitas gagal bayar (default) pada pengajuan kredit nasabah. Model ini dirancang untuk berfungsi sebagai *decision-support system* bagi analis kredit perbankan dalam meminimalkan risiko kerugian finansial (Non-Performing Loan).

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Techniques Used:** 
  * Data Wrangling (Handling Missing Values via Imputation, Outlier Removal, Duplicate Handling)
  * Feature Engineering (One-Hot Encoding, Ordinal Encoding)
  * Addressing Class Imbalance (`scale_pos_weight` formulation)
  * Stratified Train-Test Splitting

## 📊 Business Insights (Feature Importance)
Berdasarkan model AI yang telah dilatih, tiga faktor paling krusial yang menentukan apakah nasabah akan gagal bayar adalah:
1. **Loan Grade (`loan_grade`):** Kategori risiko internal bank terbukti menjadi prediktor paling akurat.
2. **Debt-to-Income Ratio (`loan_percent_income`):** Nasabah dengan persentase cicilan utang yang terlalu besar dibandingkan total gaji bulanan memiliki risiko macet yang sangat tinggi.
3. **Home Ownership - RENT (`person_home_ownership_RENT`):** Nasabah yang masih berstatus menyewa tempat tinggal menunjukkan kerentanan finansial yang lebih tinggi dibandingkan pemilik rumah.

## 📈 Model Performance (XGBoost Baseline)
Model dievaluasi menggunakan *testing data* (20% dari populasi) yang steril dari *data leakage* dan duplikasi.
* **ROC-AUC Score:** 0.9465 (Excellent)
* **Precision (Class 1 - Default):** 80% 
* **Recall (Class 1 - Default):** 79%

*Artinya, model berhasil mendeteksi 79% dari total nasabah yang benar-benar macet, dan ketika model membunyikan alarm peringatan "Gagal Bayar", tingkat akurasi tuduhan tersebut mencapai 80%.*

## ⚙️ How to Use (Inference)
Model yang telah dilatih tersimpan sebagai `xgboost_credit_risk.pkl`. Untuk melakukan pengetesan prediksi pada nasabah baru:

1. Buka terminal dan masuk ke folder `src`.
2. Jalankan script prediksi:
   ```bash
   python predict.py