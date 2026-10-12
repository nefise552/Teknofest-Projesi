import polars as pl
import lightgbm as lgb
from sklearn.model_selection import KFold
from datetime import datetime
import gc
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# --- 1. Veri Yolları ve Ayarları Tanımlama ---
DATA_PATH = "C:/Users/polat/Desktop/data"
NUM_FOLDS = 5  # K-Fold için fold sayısı
CLICKED_WEIGHT = 0.35  # "clicked" için ağırlık
ORDERED_WEIGHT = 0.65  # "ordered" için ağırlık
print("Adım 1: Veri yolları ve ayarlar tanımlandı.")

# --- 2. Veri Yükleme ---
try:
    train_sessions_path = os.path.join(DATA_PATH, "train_sessions.parquet")
    test_sessions_path = os.path.join(DATA_PATH, "test_sessions.parquet")
    content_metadata_path = os.path.join(DATA_PATH, "content", "metadata.parquet")
    content_price_data_path = os.path.join(DATA_PATH, "content", "price_rate_review_data.parquet")
    user_metadata_path = os.path.join(DATA_PATH, "user", "metadata.parquet")

    train_sessions = pl.read_parquet(train_sessions_path)
    test_sessions = pl.read_parquet(test_sessions_path)
    content_metadata = pl.read_parquet(content_metadata_path)
    content_price_data = pl.read_parquet(content_price_data_path)
    user_metadata = pl.read_parquet(user_metadata_path)
except FileNotFoundError as e:
    print(f"HATA: Dosya bulunamadı! Lütfen DATA_PATH yolunu kontrol edin. Hata veren dosya: {e.filename}")
    exit()
except OSError as e:
    print(f"HATA: Dosya yolu hatası oluştu! Lütfen DATA_PATH yolunu ve alt klasörlerin doğru olduğundan emin olun.")
    print(f"Hata detayı: {e}")
    exit()

print("Adım 2: Tüm veri dosyaları başarıyla yüklendi.")

# --- 3. Veri Birleştirme ve Ön İşleme ---
# ID'leri kategorik yapma
for col in ['user_id_hashed', 'content_id_hashed', 'session_id', 'category_id_hashed', 'brand_id_hashed',
            'merchant_id_hashed']:
    for df in [content_metadata, user_metadata, train_sessions, test_sessions, content_price_data]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))

if 'user_gender' in user_metadata.columns:
    user_metadata = user_metadata.with_columns(pl.col('user_gender').cast(pl.Categorical))

train_sessions = train_sessions.with_columns(pl.lit("train").alias("source"))
test_sessions = test_sessions.with_columns(pl.lit("test").alias("source"))
sessions = pl.concat([train_sessions, test_sessions], how="diagonal")

sessions = sessions.with_columns(pl.col("ts_hour").cast(pl.Date).alias("ts_date"))
content_price_data = content_price_data.with_columns(pl.col("update_date").cast(pl.Date).alias("ts_date"))

sessions = sessions.join(content_metadata, on="content_id_hashed", how="left")
sessions = sessions.join(user_metadata, on="user_id_hashed", how="left")
sessions = sessions.join(content_price_data, on=["content_id_hashed", "ts_date"], how="left")

print("Adım 3: Veriler birleştirildi ve ön işleme tamamlandı.")

del content_metadata, content_price_data, user_metadata, train_sessions, test_sessions
gc.collect()

# --- 4. Daha Gelişmiş Özellik Mühendisliği (Feature Engineering) ---
sessions = sessions.with_columns([
    (pl.col("original_price") - pl.col("selling_price")).alias("discount_amount"),
    ((pl.col("original_price") - pl.col("selling_price")) / pl.col("original_price").fill_null(1)).alias(
        "discount_ratio"),
    pl.col("ts_hour").dt.hour().alias("hour_of_day"),
    pl.col("ts_hour").dt.weekday().alias("day_of_week"),
    pl.col("ts_hour").dt.day().alias("day_of_month"),
    pl.col("ts_hour").dt.week().alias("week_of_year"),
    pl.col("ts_hour").dt.month().alias("month"),
    (pl.col("content_review_count") / pl.col("content_rate_count").fill_null(1)).alias("review_per_rate_ratio"),
    (pl.col("content_review_wth_media_count") / pl.col("content_review_count").fill_null(1)).alias(
        "media_review_ratio"),
    pl.col("search_term_normalized").str.len_chars().alias("search_term_length"),
])

# Kullanıcı ve ürün geçmişini sadece eğitim verisinden hesapla
train_df_for_agg = sessions.filter(pl.col("source") == "train")

# Kullanıcı etkileşim özellikleri
user_agg_features = train_df_for_agg.group_by("user_id_hashed").agg([
    pl.mean("ordered").alias("user_avg_order_rate"),
    pl.mean("clicked").alias("user_avg_click_rate"),
    pl.len().alias("user_interaction_count"),
    pl.sum("added_to_cart").alias("user_cart_count"),
    pl.sum("added_to_fav").alias("user_fav_count"),
    pl.n_unique("content_id_hashed").alias("user_unique_content_count")
])
sessions = sessions.join(user_agg_features, on="user_id_hashed", how="left")

# Ürün etkileşim özellikleri
content_agg_features = train_df_for_agg.group_by("content_id_hashed").agg([
    pl.mean("ordered").alias("content_avg_order_rate"),
    pl.mean("clicked").alias("content_avg_click_rate"),
    pl.len().alias("content_interaction_count"),
    pl.sum("added_to_cart").alias("content_cart_count"),
    pl.sum("added_to_fav").alias("content_fav_count"),
    pl.n_unique("user_id_hashed").alias("content_unique_user_count")
])
sessions = sessions.join(content_agg_features, on="content_id_hashed", how="left")

# Kullanıcı-içerik etkileşimleri
user_content_agg = train_df_for_agg.group_by(["user_id_hashed", "content_id_hashed"]).agg(
    pl.len().alias("user_content_interaction_count")
)
sessions = sessions.join(user_content_agg, on=["user_id_hashed", "content_id_hashed"], how="left")

# Boş değerleri doldurma
sessions = sessions.fill_null(0)
print("Adım 4: Özellik mühendisliği tamamlandı.")

# --- 5. Model Eğitimi için Hazırlık ---
test_df_original_categorical = sessions.filter(pl.col("source") == "test").select(["session_id", "content_id_hashed"])

categorical_cols = [col for col in sessions.columns if sessions[col].dtype == pl.Categorical]
sessions = sessions.with_columns([pl.col(c).to_physical() for c in categorical_cols])

train_df = sessions.filter(pl.col("source") == "train")
test_df = sessions.filter(pl.col("source") == "test")

features_to_drop = [
    "session_id", "user_id_hashed", "content_id_hashed", "ts_hour", "ts_date",
    "search_term_normalized", "source", "clicked", "ordered", "added_to_cart", "added_to_fav",
    "content_creation_date", "update_date", "cv_tags"
]
features_to_drop.extend([col for col in train_df.columns if "_id_hashed" in col and col not in features_to_drop])
features_to_drop.extend([col for col in train_df.columns if "_name" in col and col not in features_to_drop])

features = [col for col in train_df.columns if col not in features_to_drop]

print(f"Adım 5: Model için hazırlık tamamlandı. {len(features)} adet özellik kullanılacak.")

# --- 6. Modelleri Eğitme (K-Fold Cross-Validation ile) ---
print("\nAdım 6: Modeller K-Fold Cross-Validation ile eğitiliyor...")

X_train = train_df.select(features).to_pandas()
y_train_ordered = train_df.select("ordered").to_pandas().values.ravel()
y_train_clicked = train_df.select("clicked").to_pandas().values.ravel()

# Model parametreleri (Daha gelişmiş parametreler)
params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 3000, 'learning_rate': 0.01, 'num_leaves': 64,
    'max_depth': 8, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'class_weight': 'balanced'  # Imbalanced verisetleri için dengeleyici ağırlık
}

# K-Fold kurulumu
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)


def train_model(X_train, y_train, model_type):
    oof_preds = np.zeros(len(X_train))
    models = []

    print(f"'{model_type}' modeli eğitiliyor...")
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/{NUM_FOLDS} - '{model_type}' modeli...")
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(200, verbose=False)])

        oof_preds[val_index] = model.predict_proba(X_val)[:, 1]
        models.append(model)
        del X_tr, X_val, y_tr, y_val
        gc.collect()

    print(f"'{model_type}' AUC skoru: {roc_auc_score(y_train, oof_preds):.4f}")
    return models


models_ordered = train_model(X_train, y_train_ordered, 'ordered')
models_clicked = train_model(X_train, y_train_clicked, 'clicked')

del X_train, y_train_ordered, y_train_clicked, train_df
gc.collect()

print("Adım 6: Model eğitimi tamamlandı.")

# --- 7. Tahminleri Üretme ve Submission Dosyasını Oluşturma ---
print("\nAdım 7: Tahminler üretiliyor ve submission dosyası oluşturuluyor...")
test_df_pandas = test_df.select(features).to_pandas()
test_sessions_orig_df = test_df_original_categorical.to_pandas()

BATCH_SIZE = 500000
num_batches = int(np.ceil(len(test_df_pandas) / BATCH_SIZE))
final_predictions = []

for i in range(num_batches):
    start = i * BATCH_SIZE
    end = (i + 1) * BATCH_SIZE

    test_batch_features = test_df_pandas.iloc[start:end]
    test_batch_orig_ids = test_sessions_orig_df.iloc[start:end]

    # K-Fold modellerinin ortalama tahminlerini al
    predictions_ordered_batch = np.mean([model.predict_proba(test_batch_features)[:, 1] for model in models_ordered],
                                        axis=0)
    predictions_clicked_batch = np.mean([model.predict_proba(test_batch_features)[:, 1] for model in models_clicked],
                                        axis=0)

    batch_df = pl.DataFrame({
        "session_id": test_batch_orig_ids["session_id"],
        "content_id_hashed": test_batch_orig_ids["content_id_hashed"],
        "prediction_ordered": predictions_ordered_batch,
        "prediction_clicked": predictions_clicked_batch
    })

    final_predictions.append(batch_df)

    del test_batch_features, test_batch_orig_ids, predictions_ordered_batch, predictions_clicked_batch, batch_df
    gc.collect()

all_predictions_df = pl.concat(final_predictions)

# Ağırlıklı ortalama hesapla
all_predictions_df = all_predictions_df.with_columns(
    (pl.col("prediction_ordered") * ORDERED_WEIGHT + pl.col("prediction_clicked") * CLICKED_WEIGHT).alias("final_score")
)

all_predictions_df = all_predictions_df.sort(["session_id", "final_score"], descending=[False, True])

submission_df = all_predictions_df.group_by("session_id").agg(
    pl.col("content_id_hashed").implode().alias("prediction")
)

submission_df = submission_df.with_columns(
    pl.col("prediction").list.join(" ")
).sort("session_id")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = f"submission_v3_{timestamp}.csv"
submission_df.write_csv(submission_filename)

print(f"\nAdım 7: Submission dosyası '{submission_filename}' başarıyla oluşturuldu.")