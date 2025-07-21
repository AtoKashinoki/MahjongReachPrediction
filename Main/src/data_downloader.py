# src/data_downloader.py (完全修正版)

import os
import logging
import time
import requests
import gzip
from typing import Optional, List
from bs4 import BeautifulSoup  # BeautifulSoupをインポート

# configから設定をインポート
from config import RAW_DATA_DIR

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定数
MJV_DOMAINS = ["e.mjv.jp", "f.mjv.jp"]
HEADERS = {'User-Agent': 'Mozilla/5.0'}
FAILURE_THRESHOLD = 5


def download_xml_by_id(log_id: str, retries: int = 3, backoff_factor: float = 0.5) -> Optional[str]:
    """
    指定された牌譜IDのXMLデータをダウンロードして保存する。
    【修正】HTMLページから純粋なXMLデータを抽出するロジックを追加。
    """
    output_dir = RAW_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{log_id}.xml")

    if os.path.exists(output_path):
        logging.info(f"File already exists: {output_path}")
        return output_path

    for attempt in range(retries):
        for domain in MJV_DOMAINS:
            url = f"https://{domain}/0/log/?{log_id}"
            try:
                logging.info(f"Attempting to download from {url}")
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.raise_for_status()

                # --- 【ここからが重要な修正】 ---
                # ダウンロードした内容をHTMLとして解析
                soup = BeautifulSoup(response.content, 'html.parser')

                # <textarea id="mjlog"> タグを探す
                mjlog_textarea = soup.find('textarea', id='mjlog')

                if mjlog_textarea:
                    # textareaの中身（純粋なXMLデータ）を取得
                    xml_data = mjlog_textarea.string

                    # 取得したXMLデータをファイルに書き込む
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(xml_data)

                    logging.info(f"Successfully extracted and saved XML to {output_path}")
                    return output_path
                else:
                    logging.error(f"Could not find mjlog textarea in the response from {url}")
                    # このドメインでは失敗したので、次のドメインを試す
                    continue
                # --- 【修正ここまで】 ---

            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to download from {domain} (attempt {attempt + 1}/{retries}): {e}")
                break  # ドメインを変えても無駄そうなので、リトライ待機へ

        time.sleep(backoff_factor * (2 ** attempt))

    logging.error(f"Failed to download log {log_id} after {retries} retries.")
    return None


def _get_log_ids_from_archive(date_str: str) -> List[str]:
    """指定された日付の牌譜IDリストを取得する。"""
    # (この関数は変更なし)
    url = f"https://tenhou.net/sc/raw/dat/scc{date_str}.html.gz"
    logging.info(f"Fetching log list from: {url}")

    failure_count = 0
    while failure_count < FAILURE_THRESHOLD:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            html_body = gzip.decompress(response.content).decode('utf-8')
            soup = BeautifulSoup(html_body, 'html.parser')

            log_links = soup.find_all('a', href=lambda href: href and '?log=' in href)
            log_ids = [link['href'].split('?log=')[1].split('&ts=')[0] for link in log_links]

            logging.info(f"Found {len(log_ids)} log IDs for {date_str}.")
            return log_ids
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to fetch log list: {e}. Retrying...")
            failure_count += 1
            time.sleep(1)

    logging.error("Circuit breaker tripped: Too many consecutive failures fetching log list.")
    return []


def batch_download_from_date(date_str: str, count: int):
    """指定された日付の牌譜を指定件数ダウンロードする。"""
    # (この関数は変更なし)
    log_ids = _get_log_ids_from_archive(date_str)
    if not log_ids:
        logging.error("No log IDs found. Aborting batch download.")
        return

    for i, log_id in enumerate(log_ids[:count]):
        logging.info(f"--- Downloading [{i + 1}/{count}]: {log_id} ---")
        download_xml_by_id(log_id)
        time.sleep(1)  # サーバー負荷軽減のための待機
