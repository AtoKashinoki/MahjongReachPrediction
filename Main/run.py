# Main/run.py

import sys
import os
import logging
import argparse

# srcディレクトリをパスに追加
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(SRC_DIR)

# 必要なモジュールをインポート
import config
from data_downloader import batch_download_from_date
from feature_extractor import process_all_logs
# --- [修正点 1] 正しい関数名をインポート ---
from train import train_and_save_model
from predict import predict_ron
from utils import human_input_to_csv
from visualizer import visualize_decision_tree
from evaluator import evaluate_model

# ログ設定
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Command Handlers ---

def handle_train(args):
    """'train'コマンドのロジックを処理する。"""
    # --- [修正点 2] 引数を正しく処理し、関数に渡す ---
    # コマンドライン引数から値を取得。指定がなければconfig.pyのデフォルト値を使用。
    window_size = args.window_size if args.window_size is not None else config.FEATURE_WINDOW_SIZE
    max_depth = args.max_depth if args.max_depth is not None else config.MODEL_PARAMS['dt']['max_depth']

    if args.window_size is not None:
        logging.info(f"コマンドライン引数により、河の履歴長(window_size)を {window_size} に設定しました。")

    if args.max_depth is not None:
        if args.model_type == 'dt':
            logging.info(f"コマンドライン引数により、決定木のmax_depthを {max_depth} に設定しました。")
        else:
            logging.warning("--max_depth引数は決定木(dt)モデルでのみ有効です。この引数は無視されます。")

    logging.info(f"--- [Command: Train] モデル({args.model_type})の学習を開始します ---")
    # 正しい関数名を、必要な引数をすべて渡して呼び出す
    train_and_save_model(
        model_type=args.model_type,
        window_size=window_size,
        max_depth=max_depth
    )
    logging.info(f"--- [Command: Train] モデルの学習が完了しました ---")


def handle_batch_download(args):
    """'batch_download'コマンドのロジックを処理する。"""
    logging.info(f"--- [Command: Batch Download] {args.date} の牌譜を {args.count} 件ダウンロードします ---")
    batch_download_from_date(date_str=args.date, count=args.count)


def handle_extract(args):
    """'extract'コマンドのロジックを処理する。"""
    # extractコマンドもwindow_sizeを引数として受け取るようにする
    if args.window_size is not None:
        # configの値を直接上書きすることで、下流のprocess_all_logsに反映させる
        config.FEATURE_WINDOW_SIZE = args.window_size
        logging.info(f"コマンドライン引数により、河の履歴長(window_size)を {config.FEATURE_WINDOW_SIZE} に設定しました。")

    logging.info("--- [Command: Extract Features] 特徴量抽出を開始します ---")
    process_all_logs()


def handle_predict(args):
    """'predict'コマンドのロジックを処理する。"""
    logging.info(f"--- [Command: Predict] モデル({args.model_type})で予測を実行します ---")
    try:
        logging.info(f"入力された牌: '{args.input_data}'")

        # Note: predictコマンドはconfig.pyのwindow_sizeに依存する。
        # モデルと設定が一致しているか確認が必要。
        csv_input = human_input_to_csv(
            args.input_data,
            window_size=config.FEATURE_WINDOW_SIZE
        )
        logging.info("モデルが解釈するベクトル形式に変換しました。")
        predict_ron(model_type=args.model_type, input_data_csv=csv_input)
    except ValueError as e:
        logging.error(f"入力データのエラー: {e}")
        logging.error("入力形式を確認してください。'--help'で詳細な説明が見られます。")


def handle_visualize_tree(args):
    """'visualize_tree'コマンドのロジックを処理する。"""
    logging.info(f"--- [Command: Visualize Tree] 決定木の可視化を開始します (max_depth={args.max_depth}) ---")
    visualize_decision_tree(max_depth=args.max_depth)


def handle_evaluate(args):
    """'evaluate'コマンドのロジックを処理する。"""
    logging.info(f"--- [Command: Evaluate] モデル({args.model_type})の精度評価を開始します ---")
    evaluate_model(model_type=args.model_type)


def setup_parsers():
    """全てのコマンドライン引数のパーサーをセットアップし、ルートパーサーを返す。"""
    parser = argparse.ArgumentParser(
        description="麻雀放銃予測パイプライン",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='実行するコマンド')

    # --- 'train' command ---
    train_parser = subparsers.add_parser('train', help='モデルの学習を実行')
    train_parser.add_argument('--model_type', type=str, required=True, choices=['dt', 'lr'],
                              help="使用するモデルのタイプ")
    train_parser.add_argument('--max_depth', type=int,
                              help="決定木の最大の深さ (dtモデルのみ). config.pyの値を上書きます。")
    train_parser.add_argument('--window_size', type=int,
                              help=f"リーチ者の河の履歴長 (デフォルト: {config.FEATURE_WINDOW_SIZE})")

    # --- 'batch_download' command ---
    download_parser = subparsers.add_parser('batch_download', help='指定した日付の牌譜をダウンロード')
    download_parser.add_argument('--date', type=str, default='20240101', help='ダウンロードする牌譜の日付 (YYYYMMDD)')
    download_parser.add_argument('--count', type=int, default=100, help='ダウンロードするファイル数')

    # --- 'extract' command ---
    extract_parser = subparsers.add_parser('extract', help='特徴量抽出を実行')
    extract_parser.add_argument('--window_size', type=int,
                                help=f"リーチ者の河の履歴長 (デフォルト: {config.FEATURE_WINDOW_SIZE})")

    # --- 'predict' command ---
    predict_parser = subparsers.add_parser('predict', help='放銃確率の予測を実行')
    predict_parser.add_argument('--model_type', type=str, required=True, choices=['dt', 'lr'],
                                help="使用するモデルのタイプ")
    expected_tiles = config.FEATURE_WINDOW_SIZE + 1
    input_data_help_text = f"""予測に使用する入力データ。

【形式】
河の牌(過去{config.FEATURE_WINDOW_SIZE}枚)と捨て牌(1枚)を、スペースで区切って指定します。
合計で {expected_tiles}個の牌を指定する必要があります。

【牌の記法】
  - 萬子: 1m, 2m, ..., 9m
  - 筒子: 1p, 2p, ..., 9p
  - 索子: 1s, 2s, ..., 9s
  - 字牌: e(東), s(南), w(西), n(北), p(白), f(發), c(中)
    (大文字・小文字は区別しません)

【入力例】 (config.pyのwindow_sizeが{config.FEATURE_WINDOW_SIZE}の場合)
--input_data "1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 4p"
"""
    predict_parser.add_argument('--input_data', type=str, required=True, help=input_data_help_text)

    # --- 'visualize_tree' command ---
    visualize_parser = subparsers.add_parser('visualize_tree', help='学習済み決定木(dt)モデルの構造を可視化')
    visualize_parser.add_argument('--max_depth', type=int, default=3, help="表示する決定木の最大の深さ (デフォルト: 3)")

    # --- 'evaluate' command ---
    evaluate_parser = subparsers.add_parser('evaluate', help='テストデータセットでモデルの精度を評価')
    evaluate_parser.add_argument('--model_type', type=str, required=True, choices=['dt', 'lr'],
                                 help="評価するモデルのタイプ")

    return parser


def main():
    """コマンドライン引数を解析し、適切なハンドラに処理をディスパッチする。"""
    parser = setup_parsers()
    args = parser.parse_args()

    # コマンドディスパッチャ
    command_handlers = {
        'train': handle_train,
        'batch_download': handle_batch_download,
        'extract': handle_extract,
        'predict': handle_predict,
        'visualize_tree': handle_visualize_tree,
        'evaluate': handle_evaluate,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        logging.error(f"不明なコマンドが指定されました: {args.command}")
        parser.print_help()


if __name__ == '__main__':
    main()
