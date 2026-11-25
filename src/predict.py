import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # Menyembunyikan peringatan teknis biar rapi

# 1. Load Model yang sudah disimpan tadi
print("Memuat model XGBoost...")
model = joblib.load('../models/xgboost_credit_risk.pkl')

# 2. Buat skenario 2 nasabah fiktif (Satu kaya & aman, satu berisiko tinggi)
# Urutan kolom harus sama persis dengan 17 kolom X_train saat melatih model
data_nasabah_baru = pd.DataFrame({
    'person_age': [28, 23],
    'person_income': [85000, 32000],
    'person_emp_length': [5.0, 1.0],
    'loan_grade': [0, 4], # 0 = Grade A (Aman), 4 = Grade E (Berisiko)
    'loan_amnt': [5000, 25000],
    'loan_int_rate': [7.5, 16.5],
    'loan_percent_income': [0.06, 0.78], # 6% vs 78% dari gaji
    'cb_person_default_on_file': [0, 1], # 0 = Tidak ada riwayat buruk, 1 = Ada riwayat
    'cb_person_cred_hist_length': [4, 2],
    'person_home_ownership_OTHER': [0, 0],
    'person_home_ownership_OWN': [1, 0],  # Nasabah 1 punya rumah, Nasabah 2 nyewa
    'person_home_ownership_RENT': [0, 1],
    'loan_intent_EDUCATION': [1, 0],
    'loan_intent_HOMEIMPROVEMENT': [0, 0],
    'loan_intent_MEDICAL': [0, 1],        # Nasabah 2 minjam untuk medis
    'loan_intent_PERSONAL': [0, 0],
    'loan_intent_VENTURE': [0, 0]
})

# 3. Suruh mesin melakukan prediksi
prediksi_mutlak = model.predict(data_nasabah_baru)
probabilitas = model.predict_proba(data_nasabah_baru)[:, 1] * 100 # Jadikan persen

# 4. Tampilkan hasil layaknya sistem perbankan asli
print("\n--- HASIL PREDIKSI SISTEM ---")
for i in range(len(data_nasabah_baru)):
    status = "GAGAL BAYAR (REJECT)" if prediksi_mutlak[i] == 1 else "LANCAR (APPROVE)"
    risiko = probabilitas[i]
    print(f"Nasabah ke-{i+1}:")
    print(f"  Keputusan : {status}")
    print(f"  Risiko Macet: {risiko:.2f}%\n")