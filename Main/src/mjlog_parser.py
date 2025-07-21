# src/mjlog_parser.py
import xml.etree.ElementTree as ET
import logging
import requests
from typing import Dict, Any, List, Optional

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _parse_numeric_list(value: str) -> List[int | float]:
    """カンマ区切りの文字列を数値のリストに変換するヘルパー関数"""
    if not value:
        return []
    items = value.split(',')
    parsed_items = []
    for item in items:
        try:
            parsed_items.append(float(item) if '.' in item else int(item))
        except (ValueError, TypeError):
            logging.warning(f"Could not parse numeric value '{item}' in list '{value}'.")
    return parsed_items


def parse_mjlog_xml(xml_file_path: str) -> Optional[Dict[str, Any]]:
    """
    天鳳の牌譜XMLファイルをパースし、構造化された辞書データを返す。

    Args:
        xml_file_path: パース対象のXMLファイルパス。

    Returns:
        パースされたゲームデータを含む辞書。エラー時はNoneを返す。
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"XML ParseError in {xml_file_path}: {e}")
        return None

    game_data = {'metadata': {}, 'rounds': []}

    # プレイヤー情報
    un_tag = root.find('UN')
    if un_tag is not None:
        players = []
        dan_list = un_tag.attrib.get('dan', '').split(',')
        rate_list = un_tag.attrib.get('rate', '').split(',')
        for i in range(4):
            name_key = f'n{i}'
            if name_key in un_tag.attrib:
                try:
                    player_name = requests.utils.unquote(un_tag.attrib[name_key])
                except Exception:
                    player_name = un_tag.attrib[name_key]
                players.append({
                    'name': player_name,
                    'dan': int(dan_list[i]) if i < len(dan_list) and dan_list[i] else None,
                    'rate': float(rate_list[i]) if i < len(rate_list) and rate_list[i] else None,
                })
        game_data['metadata']['players'] = players

    # 各局の情報を処理
    for round_tag in root.findall('INIT'):
        current_round = {
            'round_info': {
                'round_num_wind': int(round_tag.attrib.get('seed', '0,0').split(',')[0]),
                'honba': int(round_tag.attrib.get('seed', '0,0,0').split(',')[1]),
                'kyotaku': int(round_tag.attrib.get('seed', '0,0,0').split(',')[2]),
                'oya': int(round_tag.attrib['oya']),
                'start_scores': _parse_numeric_list(round_tag.attrib.get('ten', '')),
                'dora_indicator': int(round_tag.attrib.get('seed', '0,0,0,0,0').split(',')[4])
            },
            'initial_hands': {i: _parse_numeric_list(round_tag.attrib.get(f'hai{i}', '')) for i in range(4)},
            'events': []
        }

        # INITタグ以降のイベントをその局のイベントとして追加
        for event_tag in round_tag.xpath('following-sibling::*'):
            # 次のINITタグが来たら、この局は終了
            if event_tag.tag == 'INIT':
                break

            event_data = {'tag': event_tag.tag, 'attrib': dict(event_tag.attrib)}
            current_round['events'].append(event_data)

        game_data['rounds'].append(current_round)

    return game_data  # src/mjlog_parser.py (修正版)


import xml.etree.ElementTree as ET
import logging
import requests
from typing import Dict, Any, List, Optional

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _parse_numeric_list(value: str) -> List[int | float]:
    """カンマ区切りの文字列を数値のリストに変換するヘルパー関数"""
    if not value:
        return []
    items = value.split(',')
    parsed_items = []
    for item in items:
        try:
            parsed_items.append(float(item) if '.' in item else int(item))
        except (ValueError, TypeError):
            logging.warning(f"Could not parse numeric value '{item}' in list '{value}'.")
    return parsed_items


def parse_mjlog_xml(xml_file_path: str) -> Optional[Dict[str, Any]]:
    """
    天鳳の牌譜XMLファイルをパースし、構造化された辞書データを返す。

    Args:
        xml_file_path: パース対象のXMLファイルパス。

    Returns:
        パースされたゲームデータを含む辞書。エラー時はNoneを返す。
    """
    try:
        # 【修正点2】FileNotFoundErrorも捕捉するように変更
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        logging.error(f"Failed to parse {xml_file_path}: {e}")
        return None

    game_data = {'metadata': {}, 'rounds': []}

    # プレイヤー情報
    un_tag = root.find('UN')
    if un_tag is not None:
        players = []
        dan_list = un_tag.attrib.get('dan', '').split(',')
        rate_list = un_tag.attrib.get('rate', '').split(',')
        for i in range(4):
            name_key = f'n{i}'
            if name_key in un_tag.attrib:
                try:
                    player_name = requests.utils.unquote(un_tag.attrib[name_key])
                except Exception:
                    player_name = un_tag.attrib[name_key]
                players.append({
                    'name': player_name,
                    'dan': int(dan_list[i]) if i < len(dan_list) and dan_list[i] else None,
                    'rate': float(rate_list[i]) if i < len(rate_list) and rate_list[i] else None,
                })
        game_data['metadata']['players'] = players

    # 【修正点1】xpathを使わないロジックに変更
    current_round = None
    for event_tag in root:
        if event_tag.tag == 'INIT':
            # 新しい局が始まったら、current_roundを初期化
            current_round = {
                'round_info': {
                    'round_num_wind': int(event_tag.attrib.get('seed', '0,0').split(',')[0]),
                    'honba': int(event_tag.attrib.get('seed', '0,0,0').split(',')[1]),
                    'kyotaku': int(event_tag.attrib.get('seed', '0,0,0').split(',')[2]),
                    'oya': int(event_tag.attrib['oya']),
                    'start_scores': _parse_numeric_list(event_tag.attrib.get('ten', '')),
                    'dora_indicator': int(event_tag.attrib.get('seed', '0,0,0,0,0').split(',')[4])
                },
                'initial_hands': {i: _parse_numeric_list(event_tag.attrib.get(f'hai{i}', '')) for i in range(4)},
                'events': []
            }
            game_data['rounds'].append(current_round)

        # INIT以外のタグで、かつ局が始まっている場合、イベントを現在の局に追加
        elif current_round is not None:
            event_data = {'tag': event_tag.tag, 'attrib': dict(event_tag.attrib)}
            current_round['events'].append(event_data)

    return game_data
