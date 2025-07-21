# src/evaluator.py

import os
import glob
import logging
import joblib
import numpy as np
from lxml import etree

from config import MODEL_PATHS, TEST_DATA_DIR, FEATURE_WINDOW_SIZE

# Player ID to draw/discard tag prefix mapping
PLAYER_TAG_PREFIXES = {
    '0': ('T', 'D'),
    '1': ('U', 'E'),
    '2': ('V', 'F'),
    '3': ('W', 'G'),
}


def _parse_numeric_tile_code(code_str: str) -> np.ndarray:
    """Converts a Tenhou numeric tile code string to a 10-dimensional vector."""
    vector = np.zeros(10)
    try:
        code = int(code_str)
        if 0 <= code < 108:  # Man, Pin, Sou
            suit_index = code // 36
            number = (code % 36) // 4 + 1
            vector[suit_index] = number
        elif 108 <= code < 136:  # Honor tiles
            jihai_index = (code - 108) // 4
            vector[3 + jihai_index] = 1.0
        else:
            logging.warning(f"Out-of-range tile code: {code_str}")
    except (ValueError, TypeError):
        logging.warning(f"Cannot parse tile code: {code_str}")
    return vector


def _remove_tiles_from_hand(hand: list, tiles_to_remove: list):
    """Helper to safely remove a list of tiles from the hand."""
    for tile in tiles_to_remove:
        try:
            hand.remove(tile)
        except ValueError:
            logging.debug(f"Attempted to remove tile {tile} not in hand. (Likely a called tile being discarded)")


def _decode_meld(meld_code: int) -> list:
    """
    Decodes the 'm' attribute for a meld (N tag) and returns the list of
    tile codes that should be removed from the player's hand.
    This logic is based on the Tenhou log specification.
    """
    # Relative position of the player from whom the tile was called
    # 0: self (Ankan), 1: shimocha, 2: toimen, 3: kamicha
    from_who_rel = (meld_code >> 0) & 0x3

    if (meld_code >> 2) & 1:  # Chi
        # The called tile is always from kamicha (from_who_rel should be 3)
        t0, t1, t2 = (meld_code >> 3) & 3, (meld_code >> 5) & 3, (meld_code >> 7) & 3
        base = (meld_code >> 10) // 3
        # The meld is composed of three tiles. One is called, two are from the hand.
        # The 'from_who_rel' value in the tenhou spec for chi is which tile in the sequence was called.
        # 0: lowest tile, 1: middle, 2: highest.
        # We need to find the other two.
        meld_tiles = [
            str(base * 4 + t0),
            str((base + 1) * 4 + t1),
            str((base + 2) * 4 + t2)
        ]
        # The called tile is determined by the relative position of the discarder (always kamicha for chi)
        # but the m-attribute encodes which tile in the sequence was called.
        # This part of the spec is confusing. A simpler approach is to find the sequence.
        # Let's find which tile was called to form the sequence.
        # The spec says bits 3,4,5 determine the called tile's position in the sequence.
        called_pos = (meld_code >> 3) & 0x7
        if called_pos == 0:  # 456, called 4
            return [meld_tiles[1], meld_tiles[2]]
        if called_pos == 1:  # 456, called 5
            return [meld_tiles[0], meld_tiles[2]]
        if called_pos == 2:  # 456, called 6
            return [meld_tiles[0], meld_tiles[1]]
        # Fallback for old spec
        del meld_tiles[from_who_rel]
        return meld_tiles


    elif (meld_code >> 3) & 1:  # Pon
        # For a pon, two identical tiles are removed from the hand.
        base = (meld_code >> 9) // 3
        # We need to find two tiles of the same kind (e.g., two 1-man, ignoring red five)
        tiles_to_remove = []
        count = 0
        for tile_in_hand in reversed(hand):  # Iterate backwards to safely remove
            if int(tile_in_hand) // 4 == base:
                tiles_to_remove.append(tile_in_hand)
                count += 1
                if count == 2:
                    return tiles_to_remove
        return tiles_to_remove  # Should be 2 tiles

    elif (meld_code >> 4) & 1:  # Chakan (Added Kan)
        # For an added kan, one tile is removed from the hand to add to an existing pon.
        base = (meld_code >> 10) // 3
        return [str(base * 4 + ((meld_code >> 5) & 3))]

    else:  # Ankan (Concealed Kan) or Daiminkan (Open Kan)
        base = (meld_code >> 8) // 2
        if from_who_rel == 0:  # Ankan
            # Four identical tiles are removed from the hand.
            tiles_to_remove = []
            count = 0
            for tile_in_hand in reversed(hand):
                if int(tile_in_hand) // 4 == base:
                    tiles_to_remove.append(tile_in_hand)
                    count += 1
                    if count == 4:
                        return tiles_to_remove
            return tiles_to_remove
        else:  # Daiminkan
            # Three identical tiles are removed from the hand.
            tiles_to_remove = []
            count = 0
            for tile_in_hand in reversed(hand):
                if int(tile_in_hand) // 4 == base:
                    tiles_to_remove.append(tile_in_hand)
                    count += 1
                    if count == 3:
                        return tiles_to_remove
            return tiles_to_remove
    return []


def _reconstruct_dealer_hand(init_tag, dealer_id, final_discard_tag):
    """
    Reconstructs the dealer's hand up to the point of dealing in,
    with robust handling for all meld types.
    """
    initial_hand_str = init_tag.get(f"hai{dealer_id}")
    if not initial_hand_str:
        return None

    hand = initial_hand_str.split(',')
    events_in_round = init_tag.xpath(
        "./following-sibling::*[count(preceding-sibling::INIT) = count(./preceding-sibling::INIT)]")

    draw_prefix, discard_prefix = PLAYER_TAG_PREFIXES[dealer_id]

    for event in events_in_round:
        if event == final_discard_tag:
            last_draw_tag = final_discard_tag.getprevious()
            if last_draw_tag is not None and last_draw_tag.tag.startswith(draw_prefix):
                hand.append(last_draw_tag.tag[1:])
            return hand

        tag = event.tag

        if tag.startswith(draw_prefix):
            hand.append(tag[1:])
        elif tag.startswith(discard_prefix):
            _remove_tiles_from_hand(hand, [tag[1:]])
        elif tag == "N" and event.get("who") == dealer_id:
            m_attr = event.get("m")
            if m_attr:
                tiles_to_remove = _decode_meld(int(m_attr), hand)
                _remove_tiles_from_hand(hand, tiles_to_remove)
    return None


def _extract_test_cases_from_log(file_path):
    """Extracts test cases (riichi-ron scenarios) from a single log file."""
    try:
        tree = etree.parse(file_path)
        test_cases = []

        for init_tag in tree.xpath("//INIT"):
            events_in_round = init_tag.xpath(
                "./following-sibling::*[count(preceding-sibling::INIT) = count(./preceding-sibling::INIT)]")

            riichi_events = [e for e in events_in_round if e.tag == "REACH" and e.get("step") == "2"]
            if not riichi_events:
                continue

            for riichi_event in riichi_events:
                riichi_player_id = riichi_event.get("who")
                events_after_reach = riichi_event.xpath("./following-sibling::*")

                for event in events_after_reach:
                    is_ron_agari = (event.tag == "AGARI" and event.get("who") != event.get("fromWho"))
                    is_round_end = (event.tag in ["RYUUKYOKU", "INIT"])

                    if is_ron_agari and event.get("who") == riichi_player_id:
                        agari_event = event
                        dealer_player_id = agari_event.get("fromWho")
                        final_discard_tag = agari_event.getprevious()

                        if final_discard_tag is None:
                            break

                        dealer_hand_codes = _reconstruct_dealer_hand(init_tag, dealer_player_id, final_discard_tag)

                        if dealer_hand_codes is None or len(dealer_hand_codes) != 14:
                            logging.warning(
                                f"  [スキップ] ファイル '{os.path.basename(file_path)}' で手牌の再構築に失敗しました (牌数: {len(dealer_hand_codes) if dealer_hand_codes else 0})。")
                            break

                        riichi_river_tags = [
                            e for e in events_in_round
                            if e.tag.startswith(PLAYER_TAG_PREFIXES[riichi_player_id][1]) and e.getroottree().getpath(
                                e) < e.getroottree().getpath(riichi_event)
                        ]
                        riichi_river = [tag.tag for tag in riichi_river_tags]

                        test_cases.append({
                            "riichi_river": riichi_river[-FEATURE_WINDOW_SIZE:],
                            "dealer_hand_codes": dealer_hand_codes,
                            "actual_deal_in_code": final_discard_tag.tag[1:]
                        })
                        # Found a valid case for this riichi, so break from the inner loop
                        break
                    elif is_round_end:
                        # Round ended before a ron occurred for this riichi
                        break
        return test_cases
    except Exception as e:
        logging.error(f"ファイル '{os.path.basename(file_path)}' の致命的な解析エラー: {e}", exc_info=True)
        return []


def evaluate_model(model_type: str):
    """Evaluates the model's prediction accuracy using the test dataset."""
    model_path = MODEL_PATHS.get(model_type)
    if not model_path or not os.path.exists(model_path):
        logging.error(f"モデルファイルが見つかりません: {model_path}")
        return

    logging.info(f"モデル '{model_path}' をロード中...")
    model = joblib.load(model_path)

    # Infer window size from the trained model
    try:
        model_n_features = model.n_features_in_
        inferred_window_size = (model_n_features // 10) - 1
        if inferred_window_size != FEATURE_WINDOW_SIZE:
            logging.warning(
                f"設定ファイル(config.py)のWINDOW_SIZE({FEATURE_WINDOW_SIZE})がモデルの学習時({inferred_window_size})と異なります。モデルの値を使用します。")
            window_size = inferred_window_size
        else:
            window_size = FEATURE_WINDOW_SIZE
    except AttributeError:
        logging.warning("モデルから学習時の特徴量を特定できませんでした。config.pyのFEATURE_WINDOW_SIZEを使用します。")
        window_size = FEATURE_WINDOW_SIZE

    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.xml"))
    if not test_files:
        logging.error(f"テストデータが見つかりません。'{TEST_DATA_DIR}' にXMLファイルを配置してください。")
        return

    total_tests = 0
    correct_predictions = 0

    logging.info(f"{len(test_files)}個のテストファイルで評価を開始します...")

    for file_path in test_files:
        test_cases = _extract_test_cases_from_log(file_path)

        for case in test_cases:
            total_tests += 1
            river_vectors = [_parse_numeric_tile_code(tile_tag[1:]) for tile_tag in case["riichi_river"]]

            best_tile_code = None
            max_probability = -1.0

            for tile_code in case["dealer_hand_codes"]:
                discard_vector = _parse_numeric_tile_code(tile_code)

                padded_river_vectors = river_vectors[-window_size:]
                if len(padded_river_vectors) < window_size:
                    padding = [np.zeros(10)] * (window_size - len(padded_river_vectors))
                    padded_river_vectors = padding + padded_river_vectors

                feature_vectors = padded_river_vectors + [discard_vector]
                feature_vector_flat = np.concatenate(feature_vectors).reshape(1, -1)

                expected_features = (window_size + 1) * 10
                if feature_vector_flat.shape[1] != model.n_features_in_:
                    logging.error(
                        f"特徴量ベクトルの長さが不正です。モデルの期待値: {model.n_features_in_}, 実際: {feature_vector_flat.shape[1]}")
                    continue

                prediction = model.predict(feature_vector_flat)[0]

                if prediction > max_probability:
                    max_probability = prediction
                    best_tile_code = tile_code

            if best_tile_code == case["actual_deal_in_code"]:
                correct_predictions += 1
                logging.info(f"  [正解] 予測: {best_tile_code}, 実際: {case['actual_deal_in_code']}")
            else:
                logging.info(f"  [不正解] 予測: {best_tile_code}, 実際: {case['actual_deal_in_code']}")

    if total_tests == 0:
        logging.warning("評価対象となる「リーチ後の放銃イベント」がテストデータ内に見つかりませんでした。")
        return

    accuracy = (correct_predictions / total_tests) * 100
    logging.info("--- 評価結果 ---")
    logging.info(f"評価した放銃イベント数: {total_tests}")
    logging.info(f"予測が正解した数: {correct_predictions}")
    logging.info(f"モデル正解率: {accuracy:.2f}%")
