# Main/src/predict.py

import os
import sys
import logging
import joblib
import numpy as np

# configモジュールをインポートするためにパスを追加
# このファイルの場所から1つ上の階層 (Main/src) を取得
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_PATH)

from config import MODEL_PATHS

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_ron(model_type: str, input_data_csv: str):
    """
    学習済みモデルをロードし、入力データに対する放銃確率を予測する。

    Args:
        model_type (str): 使用するモデルのタイプ ('dt' または 'lr')。
        input_data_csv (str): カンマ区切りの特徴量データ文字列。
    """
    model_path = MODEL_PATHS.get(model_type)
    if not model_path or not os.path.exists(model_path):
        logging.error(f"エラー: モデルファイルが見つかりません。パス: {model_path}")
        logging.error("先に 'python Main/run.py train' を実行してモデルを学習させてください。")
        return

    try:
        # 1. モデルのロード
        model = joblib.load(model_path)
        logging.info(f"モデル '{model_path}' を正常にロードしました。")

        # 2. 入力データの準備
        # CSV文字列を数値のリストに変換
        feature_list = [float(x) for x in input_data_csv.split(',')]
        # scikit-learnが期待する2D配列に変換 (1サンプル, n特徴量)
        features = np.array(feature_list).reshape(1, -1)

        # 3. 予測の実行
        prediction = model.predict(features)
        predicted_value = prediction[0]  # 予測結果は配列で返ってくるため、最初の要素を取得

        logging.info("--- 予測結果 ---")
        if model_type == 'dt':
            logging.info(f"予測された放銃確率 (MSEベース): {predicted_value:.4f}")
        elif model_type == 'lr':
            # ロジスティック回帰の場合、出力はクラスラベル (0 or 1)
            result_text = "放銃する (1)" if predicted_value == 1.0 else "放銃しない (0)"
            logging.info(f"予測クラス: {result_text}")
        logging.info("--------------------")

        return predicted_value

    except FileNotFoundError:
        logging.error(f"モデルファイルが見つかりません: {model_path}")
    except ValueError as e:
        logging.error(f"入力データの形式が正しくありません: {e}")
        logging.error(f"受け取ったデータ: {input_data_csv}")
    except Exception as e:
        logging.error(f"予測中に予期せぬエラーが発生しました: {e}")
