# Main/batch_predict.py

import sys
import os
import logging
import argparse
import joblib
import numpy as np
from datetime import datetime

# srcディレクトリをパスに追加し、既存のモジュールをインポート可能にする
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(SRC_DIR)

# 既存のモジュールをインポート
import config
from utils import human_input_to_csv

# ログ設定
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _generate_feature_names(window_size: int) -> list[str]:
    """
    特徴量ベクトルの各要素に対応する名前のリストを生成する。
    例: 'river_1_num_m', 'discard_is_e'
    """
    feature_names = []
    # 各10次元ベクトルの要素名
    dim_names = [
        "num_m", "num_p", "num_s",
        "is_e", "is_s", "is_w", "is_n",
        "is_haku", "is_hatsu", "is_chun"
    ]
    # 河の牌の特徴量名 (例: river_1_num_m)
    for i in range(window_size):
        for dim_name in dim_names:
            feature_names.append(f"river_{i + 1}_{dim_name}")
    # 捨て牌の特徴量名 (例: discard_is_e)
    for dim_name in dim_names:
        feature_names.append(f"discard_{dim_name}")
    return feature_names


def run_batch_predictions(model_type: str, test_file_path: str, output_file_path: str):
    """
    テストファイルを1行ずつ読み込み、予測結果と判断材料をファイルに出力する。
    """
    # --- 1. モデルをロードして、そのプロパティを確認する ---
    model_path = config.MODEL_PATHS.get(model_type)
    if not model_path or not os.path.exists(model_path):
        logging.error(f"モデルファイルが見つかりません: {model_path}")
        return

    # 【改善】判断材料の出力は決定木モデルのみ対応
    if model_type != 'dt':
        logging.error("特徴量重要度（判断材料）の出力は決定木(dt)モデルでのみサポートされています。")
        return

    try:
        model = joblib.load(model_path)
        logging.info(f"モデル '{os.path.basename(model_path)}' を正常にロードしました。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}")
        return

    # --- 2. モデルから正しい設定値を推論する ---
    try:
        model_n_features = model.n_features_in_
        expected_tiles = model_n_features // 10
        correct_window_size = expected_tiles - 1

        model_params = model.get_params()
        model_max_depth = model_params.get('max_depth', 'N/A')

        logging.info(f"モデルは window_size={correct_window_size}, max_depth={model_max_depth} で学習されています。")
    except AttributeError:
        logging.error("モデルから学習時の特徴量数を特定できませんでした。")
        return

    # --- 3. 特徴量名を生成 ---
    feature_names = _generate_feature_names(correct_window_size)

    if not os.path.exists(test_file_path):
        logging.error(f"テストファイルが見つかりません: {test_file_path}")
        return

    logging.info(f"--- バッチ予測を開始します ---")
    logging.info(f"テストファイル: {test_file_path}")
    logging.info(f"出力先ファイル: {output_file_path}")

    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_cases = [line.strip() for line in f if line.strip()]

        if not test_cases:
            logging.warning("テストファイルが空か、有効なテストケースが含まれていません。")
            return

        # --- 4. 出力ファイルを開き、結果を書き込む ---
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(f"### 麻雀放銃予測バッチ処理レポート ###\n")
            out_f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            out_f.write(f"使用モデル: {os.path.basename(model_path)}\n")
            out_f.write(f"テストファイル: {os.path.basename(test_file_path)}\n")
            out_f.write(f"\n--- モデル設定 ---\n")
            out_f.write(f"河の履歴長 (window_size): {correct_window_size}\n")
            out_f.write(f"決定木の深さ (max_depth): {model_max_depth}\n")
            out_f.write(f"--------------------------------------------------\n\n")

            # 各テストケースに対してループ処理
            for i, input_data in enumerate(test_cases, 1):
                out_f.write(f"--- ケース {i}/{len(test_cases)} ---\n")
                out_f.write(f"入力: {input_data}\n")

                try:
                    # 人間が読める形式からモデル入力用のベクトル(CSV文字列)に変換
                    csv_input = human_input_to_csv(
                        input_data,
                        window_size=correct_window_size
                    )
                    features = np.array([float(x) for x in csv_input.split(',')]).reshape(1, -1)

                    # 予測を実行
                    prediction = model.predict(features)[0]
                    out_f.write(f"予測された放銃確率: {prediction:.4f}\n")

                    # --- 【修正】ケースごとの判断根拠（デシジョンパス）を解析 ---
                    out_f.write("判断の根拠 (デシジョンパス):\n")

                    # decision_pathメソッドでノードの経路を取得
                    path_info = model.decision_path(features)
                    # path_infoは疎行列なので、非ゼロ要素のインデックスを取得
                    node_indices = path_info.indices

                    # 経路上の各ノード（分岐）についてループ（最後の葉ノードは除く）
                    for j in range(len(node_indices) - 1):
                        node_id = node_indices[j]

                        # 分岐に使われた特徴量の情報を取得
                        feature_idx = model.tree_.feature[node_id]
                        feature_name = feature_names[feature_idx]
                        threshold = model.tree_.threshold[node_id]

                        # 実際に入力された値
                        actual_value = features[0, feature_idx]

                        # どちらの子ノードに進んだかを判断
                        go_left = actual_value <= threshold

                        # 出力文字列を整形
                        # 例: 「- river_1_is_e <= 0.5 (実際の入力値: 0.0 -> True)」
                        out_f.write(
                            f"  - Step {j + 1}: {feature_name:<20s} <= {threshold:.2f} "
                            f"(Actual: {actual_value:.1f} -> {'True' if go_left else 'False'})\n"
                        )

                except ValueError as e:
                    out_f.write(f"エラー: {e}\n")
                except Exception as e:
                    logging.error(f"ケース {i} の処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
                    out_f.write(f"予期せぬエラーが発生しました。\n")

                out_f.write("\n")  # ケース間の区切り

    except Exception as e:
        logging.error(f"ファイルの読み書き中にエラーが発生しました: {e}", exc_info=True)

    logging.info(f"--- 全てのバッチ予測が完了しました ---")
    logging.info(f"結果は '{output_file_path}' に保存されました。")


def main():
    """
    コマンドライン引数を解析し、バッチ予測プロセスを開始する。
    """
    parser = argparse.ArgumentParser(
        description="テストファイルから複数のテストケースを読み込み、予測結果と判断材料をファイルに出力します。"
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='dt',
        choices=['dt'],
        help="予測に使用するモデルのタイプ (現在 'dt' のみ対応)"
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='Main/data/test/test.txt',
        help="テストデータが1行ずつ記述されたファイルへのパス"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join('Main', 'data', 'prediction_results'),
        help="予測結果レポートの出力先ディレクトリ"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"prediction_report_{timestamp}.txt"
    output_file_path = os.path.join(args.output_dir, output_filename)

    run_batch_predictions(args.model_type, args.test_file, output_file_path)


if __name__ == '__main__':
    main()
