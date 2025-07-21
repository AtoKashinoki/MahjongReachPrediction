# src/feature_extractor.py (The Truly Final Version)

import os
import logging
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
from typing import List, Tuple

# configから設定をインポート
from config import RAW_DATA_DIR, PROCESSED_DATA_PATH, PROCESSED_LOG_PATH, FEATURE_WINDOW_SIZE

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def tile_to_vector(tile_id: int) -> np.ndarray:
    """
    単一の牌ID（天鳳形式の数字）を10次元のベクトルに変換する。
    ベクトル形式: [萬子, 筒子, 索子, 東, 南, 西, 北, 白, 發, 中]
    """
    vector = np.zeros(10)

    # 天鳳の牌IDは通常2桁の数字 (例: 11=一萬, 34=四索, 41=東)
    # 赤ドラは 51, 52, 53 で表現される
    if tile_id in (51, 52, 53):
        # 赤ドラを通常の5に変換して処理を統一
        suit = tile_id // 10
        num = 5
    else:
        suit = tile_id // 10
        num = tile_id % 10

    if 1 <= suit <= 3:  # 数牌 (1:萬子, 2:筒子, 3:索子)
        if 1 <= num <= 9:  # 有効な数牌かチェック
            vector[suit - 1] = num
    elif suit == 4:  # 字牌 (東南西北白發中)
        # 【最終修正】無効な字牌ID (48, 49など) を無視する
        if 1 <= num <= 7:  # 有効な字牌(1-7)かチェック
            vector[num + 2] = 1.0
        # else: 無効な牌IDの場合は何もしない (ベクトルは0のまま)

    return vector


def parse_tile_id(tag_str: str) -> int:
    """XMLのタグ文字列から牌IDを抽出する。"""
    # この関数は数字を含むタグ(D11, T34など)でのみ呼ばれることを想定
    return int(''.join(filter(str.isdigit, tag_str)))


def extract_features_from_log(log_filepath: str) -> List[Tuple[np.ndarray, float]]:
    """
    単一の牌譜XMLログから、特徴量とラベルのペアを抽出する。
    ゲームの状態を追跡し、データ欠損に強いステートフルなロジックを実装。
    """
    try:
        tree = ET.parse(log_filepath)
        root = tree.getroot()
    except ET.ParseError:
        logging.warning(f"Could not parse XML: {log_filepath}")
        return []

    features_and_labels = []
    all_events = list(root)

    # 局の開始タグ(INIT)でイベントを分割
    round_start_indices = [i for i, event in enumerate(all_events) if event.tag == 'INIT']

    for i in range(len(round_start_indices)):
        start_index = round_start_indices[i]
        end_index = round_start_indices[i + 1] if i + 1 < len(round_start_indices) else len(all_events)
        round_events = all_events[start_index:end_index]

        # 1. 局内で最初にリーチが宣言されたイベントを探す
        for event_idx, event in enumerate(round_events):
            if event.tag == 'REACH' and event.get('step') == '1':
                # リーチ者のIDを安全に取得
                who_str = event.get('who')
                if not who_str or not who_str.isdigit():
                    continue  # 無効なリーチ宣言はスキップ
                riichi_declared_player = int(who_str)

                # 2. リーチ者の河を、リーチ宣言直前の状態まで構築する
                reacher_river_tiles = []

                # 親のIDを安全に取得し、最初のターンプレイヤーを設定
                oya_str = round_events[0].get('oya')
                if not oya_str or not oya_str.isdigit():
                    break  # 親が不明な局はスキップ
                turn_player = int(oya_str)

                for prev_event in round_events[:event_idx]:
                    tag = prev_event.tag
                    # ツモイベントでターンプレイヤーを更新
                    if tag.startswith(('T', 'U', 'V', 'W')):
                        turn_player = {'T': 0, 'U': 1, 'V': 2, 'W': 3}[tag[0]]
                    # 打牌イベント
                    elif tag.startswith(('D', 'E', 'F', 'G')):
                        if turn_player == riichi_declared_player:
                            try:
                                reacher_river_tiles.append(parse_tile_id(tag))
                            except ValueError:
                                pass  # IDのない不正な打牌タグは無視

                # 3. リーチ宣言後の他家の打牌をスキャンしてデータ点を生成
                turn_player_after_reach = riichi_declared_player
                for subsequent_event_idx in range(event_idx + 1, len(round_events)):
                    sub_event = round_events[subsequent_event_idx]
                    tag = sub_event.tag

                    if tag.startswith(('T', 'U', 'V', 'W')):
                        turn_player_after_reach = {'T': 0, 'U': 1, 'V': 2, 'W': 3}[tag[0]]
                    elif tag.startswith(('D', 'E', 'F', 'G')):
                        discarder_index = turn_player_after_reach
                        if discarder_index == riichi_declared_player:
                            continue

                        try:
                            discarded_tile = parse_tile_id(tag)
                        except ValueError:
                            continue  # 無効な打牌はスキップ

                        # 特徴量とラベルを生成
                        river_window = reacher_river_tiles[-FEATURE_WINDOW_SIZE:]
                        river_vectors = [tile_to_vector(t) for t in river_window]
                        padding_count = FEATURE_WINDOW_SIZE - len(river_vectors)
                        paddings = [np.zeros(10) for _ in range(padding_count)]
                        river_feature = np.concatenate(paddings + river_vectors)
                        discard_feature = tile_to_vector(discarded_tile)
                        final_feature = np.concatenate([river_feature, discard_feature])

                        label = 0.0
                        if subsequent_event_idx + 1 < len(round_events):
                            next_event = round_events[subsequent_event_idx + 1]
                            if next_event.tag == 'AGARI':
                                winner_str = next_event.get('who')
                                loser_str = next_event.get('fromWho')
                                if winner_str and winner_str.isdigit() and loser_str and loser_str.isdigit():
                                    winner = int(winner_str)
                                    loser = int(loser_str)
                                    if winner == riichi_declared_player and loser == discarder_index:
                                        label = 1.0

                        features_and_labels.append((final_feature, label))

                # この局で最初のリーチを処理したので、次の局に移る
                break

    return features_and_labels


def process_all_logs():
    """RAW_DATA_DIR内のログを処理し、特徴量データをCSVに保存します。"""
    if not os.path.exists(RAW_DATA_DIR):
        logging.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        return

    processed_files = set()
    if os.path.exists(PROCESSED_LOG_PATH):
        with open(PROCESSED_LOG_PATH, 'r') as f:
            processed_files = set(line.strip() for line in f)

    all_log_files = {f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.xml')}
    new_files_to_process = sorted(list(all_log_files - processed_files))

    if not new_files_to_process:
        logging.info("No new log files to process.")
        return

    logging.info(f"Found {len(new_files_to_process)} new log files to process.")

    all_data = []
    for i, filename in enumerate(new_files_to_process):
        if (i + 1) % 100 == 0 or i == 0:
            logging.info(f"  -> Processing file {i + 1}/{len(new_files_to_process)}: {filename}")
        filepath = os.path.join(RAW_DATA_DIR, filename)
        game_data = extract_features_from_log(filepath)
        all_data.extend(game_data)

    if not all_data:
        logging.warning(
            "No features were extracted from the new files. This might be because no matching events were found in the logs.")
        return

    features, labels = zip(*all_data)
    feature_cols = []
    for i in range(FEATURE_WINDOW_SIZE):
        for j in range(10):
            feature_cols.append(f'river_{i}_{j}')
    for j in range(10):
        feature_cols.append(f'discard_{j}')

    df_new = pd.DataFrame(list(features), columns=feature_cols)
    df_new['label'] = labels

    if os.path.exists(PROCESSED_DATA_PATH) and processed_files:
        logging.info(f"Appending {len(df_new)} new data points to existing CSV.")
        df_new.to_csv(PROCESSED_DATA_PATH, mode='a', header=False, index=False)
    else:
        logging.info(f"Creating new CSV with {len(df_new)} data points.")
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df_new.to_csv(PROCESSED_DATA_PATH, index=False)

    with open(PROCESSED_LOG_PATH, 'a') as f:
        for filename in new_files_to_process:
            f.write(f"{filename}\n")

    logging.info(
        f"Successfully processed {len(new_files_to_process)} new logs and extracted {len(df_new)} data points.")
