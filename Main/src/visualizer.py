# src/visualizer.py

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import logging

# configモジュールは、run.pyによって解決されるため、直接インポートできます
from config import MODEL_PATHS, FEATURE_WINDOW_SIZE

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _generate_feature_names() -> list:
    """
    モデルが使用する特徴量の名前リストを生成する。
    可視化の際に、どの特徴量が使われているか分かりやすくするため。
    """
    feature_names = []
    # ベクトル定義: [萬子, 筒子, 索子, 東, 南, 西, 北, 白, 發, 中]
    vector_names = ['manzu', 'pinzu', 'sozu', 'east', 'south', 'west', 'north', 'white', 'green', 'red']

    # 河の特徴量名
    for i in range(FEATURE_WINDOW_SIZE):
        for name in vector_names:
            feature_names.append(f'river_{i}_{name}')

    # 打牌の特徴量名
    for name in vector_names:
        feature_names.append(f'discard_{name}')

    return feature_names


def visualize_decision_tree(max_depth: int = 3):
    """
    学習済みの決定木モデルを読み込み、その構造を可視化してファイルに保存する。
    """
    model_path = MODEL_PATHS.get('dt')
    if not model_path or not os.path.exists(model_path):
        logging.error(f"エラー: モデルファイルが見つかりません。パス: {model_path}")
        logging.error("先に 'python Main/run.py train --model_type dt' を実行してモデルを学習させてください。")
        return

    # モデルと特徴量名をロード
    try:
        model = joblib.load(model_path)
        feature_names = _generate_feature_names()
        logging.info(f"モデル '{model_path}' を正常にロードしました。")
    except Exception as e:
        logging.error(f"モデルのロード中にエラーが発生しました: {e}")
        return

    # --- 1. テキスト形式でルールを出力 ---
    logging.info("決定木のルールをテキストファイルに出力中...")
    try:
        tree_rules = export_text(model, feature_names=feature_names)
        output_txt_path = 'decision_tree_rules.txt'
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(tree_rules)
        logging.info(f"ルールを '{output_txt_path}' に保存しました。")
    except Exception as e:
        logging.error(f"テキスト形式でのルール出力に失敗しました: {e}")

    # --- 2. グラフィカルなツリーを画像として出力 ---
    logging.info(f"決定木を画像ファイルに出力中... (max_depth={max_depth})")
    try:
        plt.figure(figsize=(40, 20))
        plot_tree(
            model,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            proportion=False,
            max_depth=max_depth,
            fontsize=10
        )
        output_png_path = 'decision_tree.png'
        plt.savefig(output_png_path, dpi=300)
        plt.close()
        logging.info(f"決定木を '{output_png_path}' に保存しました。")
        logging.info(f"注意: 木全体ではなく、深さ{max_depth}までを表示しています。")
    except Exception as e:
        logging.error(f"画像形式での決定木出力に失敗しました: {e}")
