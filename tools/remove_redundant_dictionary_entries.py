# flake8: noqa
"""
Usage: python remove_redundant_dictionary_entries.py

resources/dictionaries ディレクトリにある辞書データのうち、内蔵辞書のみ適用させた pyopenjtalk から得られる発音 or 読みと一致する単語を削除するツール。
02_ 以降の辞書データはアクセント情報が自動生成されているため、デフォルト辞書のみでも正確な読みが得られる単語は、辞書登録しない方が良いアクセントになると考えられる。
また、辞書登録せずとも形態素解析で正しい読みを取得できる単語を削除することで、辞書サイズを圧縮し、ゴミデータを除去し、辞書の精度低下を抑える目的もある。
"""

import csv
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import jaconv
import pyopenjtalk
from tqdm import tqdm


class ProcessResult(NamedTuple):
    """並列処理の結果を格納する型"""

    row: list[str]
    should_remove: bool
    removal_reason: str | None


def get_default_reading_pronunciation(text: str) -> tuple[str, str]:
    """pyopenjtalk のデフォルト辞書から読みと発音を取得する"""
    njd_features = pyopenjtalk.run_frontend(text)

    reads: list[str] = []
    prons: list[str] = []
    for n in njd_features:
        if n["pos"] == "記号":
            # r = n["string"]
            # p = n["string"]
            # 比較用に算出しているだけなので、差異が出やすい記号は読みや発音に含めない
            continue
        else:
            r = n["read"]
            p = n["pron"]
        # remove special chars
        for c in "’":
            r = r.replace(c, "")
            p = p.replace(c, "")
        reads.append(r)
        prons.append(p)
    return "".join(reads).strip(), "".join(prons).strip()


def process_row(row: list[str]) -> ProcessResult:
    """1行を処理し、削除すべきかどうかを判定する"""
    if not row:  # 空行をスキップ
        return ProcessResult(row, False, None)

    # CSV の形式は 表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音 を想定
    # : は除去してから比較
    # 事前に全角英数字に変換している
    surface = jaconv.h2z(row[0], ascii=True, digit=True).strip()
    reading = jaconv.h2z(row[11].replace(":", ""), ascii=True, digit=True).strip()
    pronunciation = jaconv.h2z(row[12].replace(":", ""), ascii=True, digit=True).strip()

    # surface がひらがな・カタカナのみで構成される場合、かつ3文字以上の場合は、pyopenjtalk が苦手とする
    # ひらがな・カタカナ単語の分かち書き強化のために意図的に残す
    if re.match(r"^[\u3040-\u309F\u30A0-\u30FF]{3,}$", surface):
        return ProcessResult(row, False, None)

    # 全角数字のみから構成される単語は読み間違いを防ぐため削除する
    if re.match(r"^[０-９]+$", surface):
        return ProcessResult(
            row, True, f"{surface} → Removed (Full-width numbers only)"
        )

    # 表層形が1文字の単語は読み間違いを防ぐため削除する
    if len(surface) == 1:
        return ProcessResult(row, True, f"{surface} → Removed (Single character)")

    # 全角英数字2文字以下の単語は読み間違いを防ぐため削除する
    if re.match(r"^[０-９Ａ-Ｚａ-ｚ]{1,2}$", surface):
        return ProcessResult(
            row, True, f"{surface} → Removed (Full-width alphanumeric ≤ 2 chars)"
        )

    # デフォルト辞書のみ適用した pyopenjtalk から読みと発音を取得
    default_reading, default_pronunciation = get_default_reading_pronunciation(surface)  # fmt: skip
    default_reading_without_special_chars = default_reading.replace("・", "").replace(
        "　", ""
    )
    default_pronunciation_without_special_chars = default_pronunciation.replace(
        "・", ""
    ).replace("　", "")

    # デフォルト辞書のみ適用した pyopenjtalk の発音と完全一致する場合は削除
    ## pyopenjtalk から取得した発音には「・」や全角スペースが含まれることがあるが、Mecab 辞書データの発音には含まれていないことが多いので、
    ## 除去した状態でも一致する場合は削除する
    if (
        default_pronunciation == pronunciation
        or default_pronunciation_without_special_chars == pronunciation
    ):
        return ProcessResult(row, True, f"{surface} → {default_pronunciation} (Pron)")

    # そうでないが、デフォルト辞書の読みと完全一致する場合は削除
    if default_reading == reading or default_reading_without_special_chars == reading:
        return ProcessResult(row, True, f"{surface} → {default_reading} (Read)")

    # CSV 側の「ハ」を「ワ」に変換した場合に一致する場合も削除
    if (
        reading.replace("ハ", "ワ") == default_reading
        or reading.replace("ハ", "ワ") == default_reading_without_special_chars
        or pronunciation.replace("ハ", "ワ") == default_pronunciation
        or pronunciation.replace("ハ", "ワ") == default_pronunciation_without_special_chars
    ):  # fmt: skip
        return ProcessResult(
            row, True, f'{surface} → {reading.replace("ハ", "ワ")} (Ha->Wa in CSV)'
        )

    # CSV 側の「ヲ」を「オ」に変換した場合に一致する場合も削除
    if (
        reading.replace("ヲ", "オ") == default_reading
        or reading.replace("ヲ", "オ") == default_reading_without_special_chars
        or pronunciation.replace("ヲ", "オ") == default_pronunciation
        or pronunciation.replace("ヲ", "オ") == default_pronunciation_without_special_chars
    ):  # fmt: skip
        return ProcessResult(
            row, True, f'{surface} → {reading.replace("ヲ", "オ")} (Wo->O in CSV)'
        )

    # CSV 側の「ヘ」を「エ」に変換した場合に一致する場合も削除
    if (
        reading.replace("ヘ", "エ") == default_reading
        or reading.replace("ヘ", "エ") == default_reading_without_special_chars
        or pronunciation.replace("ヘ", "エ") == default_pronunciation
        or pronunciation.replace("ヘ", "エ") == default_pronunciation_without_special_chars
    ):  # fmt: skip
        return ProcessResult(
            row, True, f'{surface} → {reading.replace("ヘ", "エ")} (He->E in CSV)'
        )

    return ProcessResult(row, False, None)


def process_csv_file(file_path: str) -> tuple[int, list[list[str]]]:
    """CSV ファイルを処理し、デフォルト辞書の読みと一致する行を削除する"""
    # 全行を読み込む
    with open(file_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    unique_rows: list[list[str]] = []
    removed_rows: list[list[str]] = []

    # ProcessPoolExecutor で並列処理
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_row, row) for row in rows]
        for future in tqdm(futures, desc="Processing rows", total=len(rows)):
            result = future.result()
            if result.should_remove:
                removed_rows.append(result.row)
                print(
                    f'\033[91m{result.removal_reason} | {",".join(result.row)}\033[0m'
                )
            else:
                unique_rows.append(result.row)

    # 結果を書き込む
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(unique_rows)

    # ファイルの末尾の改行を削除
    with open(file_path, "rb+") as f:
        f.seek(-1, 2)
        last_char = f.read(1)
        if last_char == b"\n":
            f.seek(-1, 2)
            f.truncate()

    return len(unique_rows), removed_rows


def print_removed_entries(file_name: str, removed_rows: list[list[str]]) -> None:
    """削除された行を赤色で出力する"""
    print(f"\nSummary of removed entries from {file_name}:")
    print("-" * shutil.get_terminal_size().columns)
    for row in removed_rows:
        print(f'\033[91m{",".join(row)}\033[0m')  # 赤色で出力
        print("-" * shutil.get_terminal_size().columns)


def remove_redundant_dictionary_entries() -> None:
    dict_dir = Path("resources/dictionaries")
    total_removed = 0

    for file_path in sorted(dict_dir.glob("*.csv")):
        # 01_ から始まる辞書は手動生成された辞書なのでスキップ
        if file_path.name.startswith("01_"):
            continue

        print(f"\nProcessing {file_path.name}...")
        original_count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
        new_count, removed_rows = process_csv_file(str(file_path))
        removed = original_count - new_count
        total_removed += removed

        print(f"Finished {file_path.name}: Removed {removed} entries")
        print_removed_entries(file_path.name, removed_rows)

    print(f"\nTotal removed entries: {total_removed}")


if __name__ == "__main__":
    remove_redundant_dictionary_entries()
