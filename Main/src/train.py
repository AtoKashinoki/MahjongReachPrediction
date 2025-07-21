# Main/src/train.py

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib
import logging
import argparse
import os

# srcディレクトリをパスに追加し、既存のモジュールをインポート可能にする
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# run.pyから呼び出されることを想定し、プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))
sys.path.append(PROJECT_ROOT)

from Main.src import config

# ログ設定
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_and_save_model(model_type: str, window_size: int, max_depth: int):
    """
    モデルを学習し、指定されたパスに保存する。
    """
    logging.info(f"--- '{model_type}' モデルの学習を開始します ---")
    logging.info(f"パラメータ: window_size={window_size}, max_depth={max_depth}")

    # データの読み込み
    try:
        data = pd.read_csv(config.PROCESSED_DATA_PATH)
        logging.info(f"学習データ数: {len(data)}件")
    except FileNotFoundError:
        logging.error(f"学習データファイルが見つかりません: {config.PROCESSED_DATA_PATH}")
        logging.error("先に 'Main/run.py extract' を実行して、学習データを作成してください。")
        return

    # 特徴量 (X) と ラベル (y) の分離
    X = data.drop('label', axis=1)
    y = data['label']

    # モデルの選択と学習
    if model_type == 'dt':
        # --- 【修正点】正しいパスからrandom_stateを取得 ---
        # model_typeに応じて設定を動的に取得することで、より柔軟な作りにします。
        random_state_value = config.MODEL_PARAMS.get(model_type, {}).get('random_state')
        if random_state_value is None:
            logging.warning(f"config.pyに {model_type} の random_state が設定されていません。")

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state_value
        )
        # ------------------------------------------------
    else:
        logging.error(f"未対応のモデルタイプです: {model_type}")
        return

    logging.info("モデルの学習中...")
    model.fit(X, y)
    logging.info("モデルの学習が完了しました。")

    # モデルの保存
    model_path = config.MODEL_PATHS.get(model_type)
    if not model_path:
        logging.error(f"設定ファイルに '{model_type}' のモデルパスが定義されていません。")
        return

    # 保存先ディレクトリがなければ作成
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"モデルを '{model_path}' に保存しました。")
    logging.info("--- 学習プロセス完了 ---")


def main():
    """
    このスクリプトが直接実行された場合の処理 (run.py経由がメイン)
    """
    parser = argparse.ArgumentParser(description="麻雀の放銃予測モデルを学習します。")
    parser.add_argument(
        '--model_type',
        type=str,
        default='dt',
        choices=['dt'],
        help="学習するモデルのタイプ ('dt' for Decision Tree)"
    )
    # config.pyからデフォルト値を取得
    default_window_size = config.FEATURE_WINDOW_SIZE
    default_max_depth = config.MODEL_PARAMS.get('dt', {}).get('max_depth')

    parser.add_argument(
        '--window_size',
        type=int,
        default=default_window_size,
        help=f"特徴量として考慮する河の履歴の長さ (デフォルト: {default_window_size})"
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=default_max_depth,
        help=f"決定木の最大の深さ (デフォルト: {default_max_depth})"
    )
    args = parser.parse_args()

    train_and_save_model(args.model_type, args.window_size, args.max_depth)


if __name__ == '__main__':
    main()
