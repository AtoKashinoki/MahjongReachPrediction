# Main/src/utils.py

import numpy as np
import logging


def _parse_single_tile(tile_str: str) -> np.ndarray:
    """
    単一の牌表現文字列（例: "1m", "3s", "E", "東"）を10次元ベクトルに変換する。
    Kanji and English abbreviations are supported for honor tiles.
    """
    vector = np.zeros(10)
    s = tile_str.strip().lower()
    if not s:
        return vector

    # ベクトル定義: [萬子, 筒子, 索子, 東, 南, 西, 北, 白, 發, 中]
    #               0    1    2    3    4    5    6    7    8    9

    # --- 【ここから修正】 ---
    # 字牌の処理: 英語と漢字の両方に対応
    jihai_map = {
        'e': 3, '東': 3,
        's': 4, '南': 4,
        'w': 5, '西': 5,
        'n': 6, '北': 6,
        'p': 7, '白': 7,
        'f': 8, '發': 8,
        'c': 9, '中': 9
    }
    # --- 【修正ここまで】 ---

    if s in jihai_map:
        vector[jihai_map[s]] = 1.0
        return vector

    # 数牌の処理
    try:
        num = int(s[:-1])
        suit = s[-1]
        suit_map = {'m': 0, 'p': 1, 's': 2}

        if suit in suit_map and 1 <= num <= 9:
            vector[suit_map[suit]] = num
        else:
            logging.warning(f"無効な数牌の指定です: '{tile_str}'。ゼロベクトルを返します。")
    except (ValueError, IndexError):
        logging.warning(f"解釈できない牌の指定です: '{tile_str}'。ゼロベクトルを返します。")

    return vector


def human_input_to_csv(human_str: str, window_size: int) -> str:
    """
    スペース区切りの牌文字列を、モデル入力用のCSV文字列に変換する。

    Args:
        human_str (str): "1m 2p E 7s..." のようなスペース区切りの文字列。
        window_size (int): 検証に使う、期待される河の牌の数。

    Returns:
        str: "1,0,0...,0,2,0...,1,0..." のようなCSV文字列。
    """
    tiles = human_str.split()

    expected_tiles = window_size + 1
    if len(tiles) != expected_tiles:
        raise ValueError(
            f"牌の数が正しくありません。現在の設定では河の{window_size}牌 + 捨て牌1牌 = 計{expected_tiles}個の牌が必要ですが、{len(tiles)}個指定されました。"
        )

    # 最後の牌が捨て牌、それ以外が河の牌
    river_tiles = tiles[:window_size]
    discard_tile = tiles[window_size]

    all_vectors = []

    # 河の牌をベクトル化
    for tile in river_tiles:
        all_vectors.append(_parse_single_tile(tile))

    # 捨て牌をベクトル化
    all_vectors.append(_parse_single_tile(discard_tile))

    # 全てのベクトルを連結し、1つのフラットな配列にする
    feature_vector = np.concatenate(all_vectors)

    # CSV文字列に変換
    return ",".join(map(str, feature_vector))
