# src/config.py

import os

# --- 基本的なパス設定 ---
# このファイルの場所から2つ上の階層 (Mainディレクトリ) をプロジェクトルートとする
# .../Main/src/config.py -> .../Main/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# データディレクトリ (Main/data)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 生データの保存場所 (Main/data/raw/tenhou_mjlogs_raw)
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "tenhou_mjlogs_raw")

# テストデータの保存場所 (Main/data/test)
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

# 加工済みデータの保存場所 (Main/data/processed/features_and_labels.csv)
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "features_and_labels.csv")

# 処理済みファイル名を記録するログのパス
PROCESSED_LOG_PATH = os.path.join(DATA_DIR, "processed", "processed_files.log")

# --- モデル関連のパス設定 ---
# (Main/models)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# モデルの種類ごとにパスを管理する辞書
MODEL_PATHS = {
    'lr': os.path.join(MODEL_DIR, "reach_predictor_lr_reg.joblib"),  # Linear Regression
    'dt': os.path.join(MODEL_DIR, "reach_predictor_dt_reg.joblib")  # Decision Tree Regressor
}

# 後方互換性のためのデフォルトパス
MODEL_PATH = MODEL_PATHS['dt']

# --- 特徴量エンジニアリングに関する設定 ---
# リーチ者の河を何牌まで遡って特徴量にするか
FEATURE_WINDOW_SIZE = 12

# --- モデルの学習に関する設定 ---
MODEL_PARAMS = {
    'dt': {
        'max_depth': 13,  # 決定木のデフォルトの深さ
        'random_state': 42
    },
    'lr': {
        # ロジスティック回帰用のパラメータが必要な場合はここに追加
    }
}
