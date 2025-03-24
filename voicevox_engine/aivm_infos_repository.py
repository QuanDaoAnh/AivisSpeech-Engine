# flake8: noqa: E266, B950

import asyncio
import glob
import hashlib
import sys
import threading
import time
import json
from pathlib import Path
from typing import Final, BinaryIO

import aivmlib
import httpx
from aivmlib.schemas.aivm_manifest import ModelArchitecture
from pydantic import TypeAdapter
from semver.version import Version

from voicevox_engine.logging import logger
from voicevox_engine.metas.Metas import (
    Speaker,
    SpeakerInfo,
    SpeakerStyle,
    SpeakerSupportedFeatures,
    StyleId,
    StyleInfo,
)
from voicevox_engine.model import AivmInfo, LibrarySpeaker
from voicevox_engine.utility.path_utility import get_save_dir
from voicevox_engine.utility.user_agent_utility import generate_user_agent

__all__ = ["AivmInfosRepository"]

AivmInfosCache = TypeAdapter(dict[str, AivmInfo])


class AivmInfosRepository:
    """
    インストール済み音声合成モデルのスキャンとメタデータの取得、キャッシュの管理を行うリポジトリ。
    エンジン起動時はキャッシュがあれば読み込み、バックグラウンドですべてのインストール済み音声合成モデルの情報を構築する。
    """

    # AivisHub API のベース URL
    AIVISHUB_API_BASE_URL: Final[str] = "https://api.aivis-project.com/v1"

    # AivisSpeech でサポートされているマニフェストバージョン
    SUPPORTED_MANIFEST_VERSIONS: Final[list[str]] = ["1.0"]

    # AivisSpeech でサポートされている音声合成モデルのアーキテクチャ
    SUPPORTED_MODEL_ARCHITECTURES: Final[list[ModelArchitecture]] = [
        ModelArchitecture.StyleBertVITS2,
        ModelArchitecture.StyleBertVITS2JPExtra,
    ]

    # エンジン起動の高速化に用いるキャッシュファイルの保存パス
    CACHE_FILE_PATH: Final[Path] = get_save_dir() / "aivm_infos_cache.json"

    def __init__(self, installed_models_dir: Path) -> None:
        """
        AivmInfosRepository のコンストラクタ

        Parameters
        ----------
        installed_models_dir : Path
            AIVMX ファイルのインストール先ディレクトリ
        """

        self.installed_models_dir = installed_models_dir
        self._lock = threading.Lock()

        # pytest から実行されているかどうか
        self._is_pytest = "pytest" in sys.argv[0] or "py.test" in sys.argv[0]

        # すべてのインストール済み音声合成モデルの情報を保持するマップ
        self._installed_aivm_infos: dict[str, AivmInfo] | None = None

        # コンストラクタ初期化時（＝エンジン起動時）、キャッシュがあればそこから即座に読み込む
        ## update_repository() は比較的実行コストが高い（モデル数が増えるほど時間がかかる）ため、
        ## 現在起動時に残したキャッシュを活用し、エンジンの起動を高速化する
        result = self._load_from_cache()
        if result is True:
            # キャッシュ情報が存在する際は、バックグラウンドでスキャンを開始
            def update_repository_in_background():
                # E2E テスト実行時に毎回スキャンするとあまりに時間がかかりすぎるため無効化
                if self._is_pytest:
                    return
                # BERT モデルのロードと並行させるため、少し待ってから実行
                ## aivmlib.read_aivmx_metadata() はモデルファイルをロードする関係で若干 CPU-bound だが、
                ## Python は GIL の制約により、CPU-bound な処理はマルチスレッドでもほとんど並列化できない
                ## そこで StyleBertVITS2TTSEngine 初期化時の BERT モデルのロードが GIL 外のネイティブコードで行われるのを活用し、
                ## GIL の制約を受けない BERT モデルのロードと並行させ、エンジン起動時間を短縮している
                time.sleep(1)
                try:
                    self.update_repository()
                except Exception as ex:
                    logger.error(
                        "Failed to update repository in background:", exc_info=ex
                    )

            threading.Thread(
                target=update_repository_in_background, daemon=True
            ).start()
        else:
            # キャッシュ情報が存在しない際はサーバー起動前に情報準備が必要なため、同期的にスキャンを行う
            self.update_repository()

        # この時点で確実に self._installed_aivm_infos が None でないことを保証する
        assert self._installed_aivm_infos is not None

    def get_installed_aivm_infos(self) -> dict[str, AivmInfo]:
        """
        すべてのインストール済み音声合成モデルの情報を取得する

        Returns
        -------
        aivm_infos : dict[str, AivmInfo]
            インストール済み音声合成モデルの情報 (キー: 音声合成モデルの UUID, 値: AivmInfo)
        """

        # コンストラクタ初期化時に確認した通り、この時点で self._installed_aivm_infos が None でないことを保証する
        assert self._installed_aivm_infos is not None
        return self._installed_aivm_infos

    @staticmethod
    def read_aivm_metadata(aivm_file: BinaryIO) -> aivmlib.AivmMetadata:
        """
        AIVM ファイルから AIVM メタデータを読み込む

        Args:
            aivm_file (BinaryIO): AIVM ファイル

        Returns:
            AivmMetadata: AIVM メタデータ

        Raises:
            AivmValidationError: AIVM ファイルのフォーマットが不正・AIVM メタデータのバリデーションに失敗した場合
        """

        # 引数として受け取った BinaryIO のカーソルを先頭にシーク
        aivm_file.seek(0)

        # ファイルの内容を読み込む
        header_size = int.from_bytes(aivm_file.read(8), 'little')

        # 引数として受け取った BinaryIO のカーソルを再度先頭に戻す
        aivm_file.seek(8)

        # ヘッダー部分を抽出
        header_bytes = aivm_file.read(header_size)
        try:
            header_text = header_bytes.decode('utf-8')
            header_json = json.loads(header_text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise aivmlib.AivmValidationError('Failed to decode AIVM metadata. This file is not an AIVM (Safetensors) file.')

        # "__metadata__" キーから AIVM メタデータを取得
        raw_metadata = header_json.get('__metadata__')

        # バリデーションを行った上で、AivmMetadata オブジェクトを構築して返す
        return aivmlib.validate_aivm_metadata(raw_metadata)
    
    def update_model_load_state(self, aivm_uuid: str, is_loaded: bool) -> None:
        """
        音声合成モデルのロード状態を更新する
        このメソッドは StyleBertVITS2TTSEngine 上でロード/アンロードが行われた際に呼び出される

        Parameters
        ----------
        aivm_uuid : str
            AIVM マニフェスト記載の UUID
        is_loaded : bool
            モデルがロードされているかどうか
        """

        if (
            self._installed_aivm_infos is not None
            and aivm_uuid in self._installed_aivm_infos
        ):
            self._installed_aivm_infos[aivm_uuid].is_loaded = is_loaded

    def update_repository(self) -> None:
        """
        すべてのインストール済み音声合成モデルから AIVM メタデータを取得し、内部状態を最新の情報に更新する。
        更新後の情報は次回起動時に利用するキャッシュにも反映される。
        """

        # ファイルシステムをスキャンして最新の音声合成モデルの情報を取得
        new_installed_aivm_infos = self._scan_models(self.installed_models_dir)

        # 情報の更新前に、現在保持されている既存のロード状態を新しい AivmInfo に移行する
        if self._installed_aivm_infos is not None:
            for aivm_uuid, aivm_info in new_installed_aivm_infos.items():
                if aivm_uuid in self._installed_aivm_infos:
                    aivm_info.is_loaded = self._installed_aivm_infos[aivm_uuid].is_loaded  # fmt: skip

        # 内部状態を更新
        self._installed_aivm_infos = new_installed_aivm_infos

        # AivisHub API からインストール済み音声合成モデルのアップデート情報を取得し、内部状態を更新
        try:
            self._installed_aivm_infos = asyncio.run(
                self._update_latest_version_info(self._installed_aivm_infos)
            )
        except Exception as ex:
            # AivisHub API からの情報取得に失敗しても起動に影響を与えないよう、ログ出力のみ行う
            logger.warning(
                "Failed to fetch update information. Continuing with cached model data:",
                exc_info=ex,
            )

        # 現在保持している情報をキャッシュに保存
        self._persist_to_cache()

    def _load_from_cache(self) -> bool:
        """
        キャッシュファイルからすべてのインストール済み音声合成モデルの情報を取得し、
        内部で保持している self._installed_aivm_infos に格納する。

        Returns
        -------
        bool
            キャッシュの読み込みに成功したかどうか
        """

        # キャッシュファイルが存在しない（初回起動時など）
        if not self.CACHE_FILE_PATH.exists():
            logger.info("Cache file not found, will load models directly.")
            return False

        with self._lock:
            try:
                # キャッシュファイルからインストール済みの音声合成モデルの情報を読み込む
                with open(self.CACHE_FILE_PATH, mode="r", encoding="utf-8") as f:
                    self._installed_aivm_infos = AivmInfosCache.validate_json(f.read())
                logger.info(
                    f"Loaded {len(self._installed_aivm_infos)} models from cache."
                )
                return True
            except Exception as ex:
                logger.warning("Failed to load cache file:", exc_info=ex)
                return False

    def _persist_to_cache(self) -> None:
        """
        内部で保持しているすべてのインストール済み音声合成モデルの情報を、キャッシュファイルに保存・反映する。
        """

        # まだインストール済みの音声合成モデルをスキャンし終わっていないため何も実行しない
        if self._installed_aivm_infos is None:
            return

        # 万が一保存先ディレクトリが存在しない場合は作成
        self.CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            try:
                # 一時ファイルに書き込んでから名前変更することで、
                # 書き込み中にクラッシュしてもキャッシュファイルが壊れないようにする
                temp_path = self.CACHE_FILE_PATH.with_suffix(".tmp")
                with open(temp_path, mode="w", encoding="utf-8") as f:
                    f.write(
                        AivmInfosCache.dump_json(
                            self._installed_aivm_infos, indent=4
                        ).decode("utf-8")
                    )
                # ファイル名を変更（既存のファイルは上書き）
                temp_path.replace(self.CACHE_FILE_PATH)
            except Exception as ex:
                logger.warning("Failed to save cache file:", exc_info=ex)

    @classmethod
    def _scan_models(cls, installed_models_dir: Path) -> dict[str, AivmInfo]:
        """
        指定されたディレクトリに保存されている *.aivmx ファイルを走査し、すべての音声合成モデルの情報を取得する。

        Parameters
        ----------
        installed_models_dir : Path
            AIVMX ファイルのインストール先ディレクトリ

        Returns
        -------
        dict[str, AivmInfo]
            スキャンで検出された、インストール済みのすべての音声合成モデルの情報
        """

        logger.info("Scanning installed models ...")
        start_time = time.time()

        # AIVMX ファイルのインストール先ディレクトリ内に配置されている .aivmx ファイルのパスを取得
        aivm_file_paths = glob.glob(str(installed_models_dir / "*.aivmx")) + glob.glob(str(installed_models_dir / "*.aivm"))

        # 各 AIVMX ファイルごとに
        aivm_infos: dict[str, AivmInfo] = {}
        for aivm_file_path in aivm_file_paths:

            # 最低限のパスのバリデーション
            aivm_file_path = Path(aivm_file_path)
            if not aivm_file_path.exists():
                logger.warning(f"{aivm_file_path}: File not found.")
                continue
            if not aivm_file_path.is_file():
                logger.warning(f"{aivm_file_path}: Not a file.")
                continue

            # AIVM メタデータの読み込み
            try:
                with open(aivm_file_path, mode="rb") as f:
                    if aivm_file_path.suffix == ".aivmx":
                        aivm_metadata = aivmlib.read_aivmx_metadata(f)
                        aivm_manifest = aivm_metadata.manifest
                    else:
                        aivm_metadata = AivmInfosRepository.read_aivm_metadata(f)
                        aivm_manifest = aivm_metadata.manifest
            except aivmlib.AivmValidationError as ex:
                logger.warning(
                    f"{aivm_file_path}: Failed to read AIVM/AIVMX metadata:",
                    exc_info=ex,
                )
                continue

            # 音声合成モデルの UUID
            aivm_uuid = str(aivm_manifest.uuid)

            # すでに同一 UUID のファイルがインストール済みかどうかのチェック
            if aivm_uuid in aivm_infos:
                logger.info(
                    f"{aivm_file_path}: AIVM model {aivm_uuid} is already installed."
                )
                continue

            # マニフェストバージョンのバリデーション
            # バージョン文字列をメジャー・マイナーに分割
            manifest_version_parts = aivm_manifest.manifest_version.split(".")
            if len(manifest_version_parts) != 2:
                logger.warning(
                    f"{aivm_file_path}: Invalid AIVM manifest version format: {aivm_manifest.manifest_version}"
                )
                continue
            # サポートされているマニフェストバージョンごとにチェック
            manifest_major, _ = map(int, manifest_version_parts)
            for supported_manifest_version in cls.SUPPORTED_MANIFEST_VERSIONS:
                # メジャーバージョンを取得
                supported_major = int(supported_manifest_version.split(".")[0])
                if manifest_major != supported_major:
                    # メジャーバージョンが AIVM マニフェストのものと異なる場合はスキップ
                    logger.warning(
                        f"{aivm_file_path}: AIVM manifest version {aivm_manifest.manifest_version} is not supported (different major version)."
                    )
                    continue
                # 同じメジャーバージョンだが、より新しいマイナーバージョンの場合は警告を出して続行
                elif (
                    aivm_manifest.manifest_version
                    not in cls.SUPPORTED_MANIFEST_VERSIONS
                ):
                    logger.warning(
                        f"{aivm_file_path}: AIVM manifest version {aivm_manifest.manifest_version} is newer than supported versions. Trying to load anyway..."
                    )

            # 音声合成モデルのアーキテクチャがサポートされているかどうかのチェック
            if aivm_manifest.model_architecture not in cls.SUPPORTED_MODEL_ARCHITECTURES:  # fmt: skip
                logger.warning(
                    f"{aivm_file_path}: Model architecture {aivm_manifest.model_architecture} is not supported."
                )
                continue

            # 仮の AivmInfo モデルを作成
            aivm_info = AivmInfo(
                is_loaded=False,
                is_update_available=False,
                latest_version=aivm_manifest.version,  # 初期値として AIVM マニフェスト記載のバージョンを設定
                # AIVMX ファイルのインストール先パス
                file_path=aivm_file_path,
                # AIVMX ファイルのインストールサイズ (バイト単位)
                file_size=aivm_file_path.stat().st_size,
                # AIVM マニフェスト
                manifest=aivm_manifest,
                # 話者情報は後で追加するため、空リストを渡す
                speakers=[],
            )

            # 話者情報を LibrarySpeaker に変換し、AivmInfo.speakers に追加
            for speaker_manifest in aivm_manifest.speakers:
                speaker_uuid = str(speaker_manifest.uuid)

                # AivisSpeech Engine は日本語のみをサポートするため、日本語をサポートしない話者は除外
                ## 念のため小文字に変換してから比較
                supported_langs = [
                    lang.lower() for lang in speaker_manifest.supported_languages
                ]
                if not any(lang in supported_langs for lang in ['ja', 'ja-jp']):  # fmt: skip
                    logger.warning(f"{aivm_file_path}: Speaker {speaker_uuid} does not support Japanese. Ignoring.")  # fmt: skip
                    continue

                # 話者アイコンを Base64 文字列に変換
                speaker_icon = cls.extract_base64_from_data_url(speaker_manifest.icon)

                # スタイルごとのメタデータを取得
                speaker_styles: list[SpeakerStyle] = []
                style_infos: list[StyleInfo] = []
                for style_manifest in speaker_manifest.styles:

                    # AIVM マニフェスト内の話者スタイル ID を VOICEVOX ENGINE 互換の StyleId に変換
                    style_id = cls.local_style_id_to_style_id(style_manifest.local_id, speaker_uuid)  # fmt: skip

                    # SpeakerStyle の作成
                    speaker_style = SpeakerStyle(
                        # VOICEVOX ENGINE 互換のスタイル ID
                        id=style_id,
                        # スタイル名
                        name=style_manifest.name,
                        # AivisSpeech は歌唱音声合成に対応しないので talk で固定
                        type="talk",
                    )
                    speaker_styles.append(speaker_style)

                    # StyleInfo の作成
                    style_info = StyleInfo(
                        # VOICEVOX ENGINE 互換のスタイル ID
                        id=style_id,
                        # アイコン画像
                        ## 未指定時は話者のアイコン画像がスタイルのアイコン画像として使われる
                        icon=cls.extract_base64_from_data_url(style_manifest.icon) if style_manifest.icon else speaker_icon,
                        # 立ち絵を省略
                        ## VOICEVOX ENGINE 本家では portrait に立ち絵が入るが、AivisSpeech Engine では敢えてアイコン画像のみを設定する
                        portrait=None,
                        # ボイスサンプル
                        voice_samples=[
                            cls.extract_base64_from_data_url(sample.audio)
                            for sample in style_manifest.voice_samples
                        ],
                        # 書き起こしテキスト
                        voice_sample_transcripts=[
                            sample.transcript
                            for sample in style_manifest.voice_samples
                        ],
                    )  # fmt: skip
                    style_infos.append(style_info)

                # LibrarySpeaker の作成
                ## 事前に取得・生成した SpeakerStyle / StyleInfo をそれぞれ Speaker / SpeakerInfo に設定する
                aivm_info_speaker = LibrarySpeaker(
                    # 話者情報
                    speaker=Speaker(
                        # 話者 UUID
                        speaker_uuid=speaker_uuid,
                        # 話者名
                        name=speaker_manifest.name,
                        # 話者のバージョン
                        ## 音声合成モデルのバージョンを話者のバージョンとして設定する
                        version=aivm_manifest.version,
                        # AivisSpeech Engine では全話者に対し常にモーフィング機能を無効化する
                        ## Style-Bert-VITS2 の仕様上音素長を一定にできず、話者ごとに発話タイミングがずれてまともに合成できないため
                        supported_features=SpeakerSupportedFeatures(
                            permitted_synthesis_morphing="NOTHING",
                        ),
                        # 話者スタイル情報
                        styles=speaker_styles,
                    ),
                    # 追加の話者情報
                    speaker_info=SpeakerInfo(
                        # ライセンス (Markdown またはプレーンテキスト)
                        ## 同一 AIVM / AIVMX ファイル内のすべての話者は同一のライセンスを持つ
                        policy=aivm_manifest.license if aivm_manifest.license else "",
                        # アイコン画像
                        ## VOICEVOX ENGINE 本家では portrait に立ち絵が入るが、AivisSpeech Engine では敢えてアイコン画像を設定する
                        portrait=speaker_icon,
                        # 追加の話者スタイル情報
                        style_infos=style_infos,
                    ),
                )  # fmt: skip
                aivm_info.speakers.append(aivm_info_speaker)

            # 完成した AivmInfo を UUID をキーとして追加
            aivm_infos[aivm_uuid] = aivm_info

        # 音声合成モデル名でソートしてから返す
        sorted_aivm_infos = dict(sorted(aivm_infos.items(), key=lambda x: x[1].manifest.name))  # fmt: skip
        logger.info(
            f"Scanned {len(sorted_aivm_infos)} installed models. ({time.time() - start_time:.2f}s)"
        )

        return sorted_aivm_infos

    @classmethod
    async def _update_latest_version_info(
        cls, aivm_infos: dict[str, AivmInfo]
    ) -> dict[str, AivmInfo]:
        """
        指定された音声合成モデルのアップデート情報を取得し、AivmInfo の latest_version と is_update_available を更新した上で返す。
        音声合成モデルごとに並行して HTTP API を叩くことで高速化を図っている（このため非同期メソッドとして実装している）。

        Parameters
        ----------
        aivm_infos : dict[str, AivmInfo]
            スキャンで検出された、インストール済みのすべての音声合成モデルの情報

        Returns
        -------
        dict[str, AivmInfo]
            AivisHub 上のアップデート情報が反映された、インストール済みのすべての音声合成モデルの情報
        """

        async def fetch_latest_version(aivm_info: AivmInfo) -> None:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{cls.AIVISHUB_API_BASE_URL}/aivm-models/{aivm_info.manifest.uuid}",
                        headers={"User-Agent": generate_user_agent()},
                        timeout=5.0,  # 5秒でタイムアウト
                    )

                    # 404 の場合は AivisHub に公開されていないモデルのためスキップ
                    if response.status_code == 404:
                        return

                    # 200 以外のステータスコードは異常なのでエラーとして処理
                    if response.status_code != 200:
                        logger.warning(
                            f"Failed to fetch model info for {aivm_info.manifest.uuid} from AivisHub (HTTP Error {response.status_code})."
                        )
                        logger.warning(f"Response: {response.text}")
                        return

                    model_info = response.json()

                    # model_files から最新の AIVMX ファイルのバージョンを取得
                    latest_aivmx_version = next(
                        (
                            file
                            for file in model_info["model_files"]
                            if file["model_type"] == "AIVMX"
                        ),
                        None,
                    )

                    if latest_aivmx_version is not None:
                        # latest_version を更新
                        aivm_info.latest_version = latest_aivmx_version["version"]
                        # バージョン比較を行い is_update_available を更新
                        current_version = Version.parse(aivm_info.manifest.version)
                        latest_version = Version.parse(aivm_info.latest_version)
                        aivm_info.is_update_available = latest_version > current_version

            except httpx.TimeoutException:
                logger.warning(
                    f"Timeout while fetching model info for {aivm_info.manifest.uuid} from AivisHub."
                )
            except Exception as ex:
                # エラーが発生しても起動に影響を与えないよう、ログ出力のみ行う
                # - httpx.RequestError: ネットワークエラーなど
                # - KeyError: レスポンスの JSON に必要なキーが存在しない
                # - StopIteration: model_files に AIVMX が存在しない
                # - ValueError: Version.parse() が失敗
                logger.warning(
                    f"Failed to fetch model info for {aivm_info.manifest.uuid} from AivisHub:",
                    exc_info=ex,
                )

        # 全モデルの更新タスクを作成
        update_tasks = [
            fetch_latest_version(aivm_info) for aivm_info in aivm_infos.values()
        ]

        # 全タスクを同時に実行
        await asyncio.gather(*update_tasks, return_exceptions=True)
        logger.info("Updated latest version information.")

        # 新バージョンの更新がある場合はログに出力
        update_available_models = [
            aivm_info
            for aivm_info in aivm_infos.values()
            if aivm_info.is_update_available
        ]
        if len(update_available_models) > 0:
            logger.info(f"Update available {len(update_available_models)} models:")
            for aivm_info in update_available_models:
                logger.info(
                    f"- {aivm_info.manifest.name} ({aivm_info.manifest.uuid}) v{aivm_info.manifest.version} -> v{aivm_info.latest_version}"
                )

        return aivm_infos

    @staticmethod
    def extract_base64_from_data_url(data_url: str) -> str:
        """
        Data URL から Base64 部分のみを取り出す

        Parameters
        ----------
        data_url : str
            Data URL

        Returns
        -------
        base64 : str
            Base64 部分
        """

        # バリデーション
        if not data_url:
            raise ValueError("Data URL is empty.")
        if not data_url.startswith("data:"):
            raise ValueError("Invalid data URL format.")

        # Data URL のプレフィックスを除去して、カンマの後の Base64 エンコードされた部分を取得
        if "," in data_url:
            base64_part = data_url.split(",", 1)[1]
        else:
            raise ValueError("Invalid data URL format.")
        return base64_part

    @staticmethod
    def local_style_id_to_style_id(local_style_id: int, speaker_uuid: str) -> StyleId:
        """
        AIVM マニフェスト内のローカルなスタイル ID を VOICEVOX ENGINE 互換のグローバルに一意な StyleId に変換する

        Parameters
        ----------
        local_style_id : int
            AIVM マニフェスト内のローカルなスタイル ID
        speaker_uuid : str
            話者の UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        style_id : StyleId
            VOICEVOX ENGINE 互換のグローバルに一意なスタイル ID
        """

        # AIVM マニフェスト内のスタイル ID は、話者ごとにローカルな 0 から始まる連番になっている
        # この値は config.json に記述されているハイパーパラメータの data.style2id の値と一致する
        # 一方 VOICEVOX ENGINE は互換性問題？による歴史的経緯でスタイル ID のみを音声合成 API に渡す形となっており、
        # スタイル ID がグローバルに一意になっていなければならない
        # そこで、話者の UUID とローカルなスタイル ID を組み合わせて、
        # グローバルに一意なスタイル ID (符号付き 32bit 整数) に変換する

        MAX_UUID_BITS = 27  # UUID のハッシュ値の bit 数
        UUID_BIT_MASK = (1 << MAX_UUID_BITS) - 1  # 27bit のマスク
        LOCAL_STYLE_ID_BITS = 5  # ローカルスタイル ID の bit 数
        LOCAL_STYLE_ID_MASK = (1 << LOCAL_STYLE_ID_BITS) - 1  # 5bit のマスク
        SIGN_BIT = 1 << 31  # 32bit 目の符号 bit

        if not speaker_uuid:
            raise ValueError("speaker_uuid must be a non-empty string")
        if not (0 <= local_style_id <= 31):
            raise ValueError("local_style_id must be an integer between 0 and 31")

        # UUID をハッシュ化し、27bit 整数に収める
        uuid_hash = int(hashlib.md5(speaker_uuid.encode(), usedforsecurity=False).hexdigest(), 16) & UUID_BIT_MASK  # fmt: skip
        # ローカルスタイル ID を 0 から 31 の範囲に収める
        local_style_id_masked = local_style_id & LOCAL_STYLE_ID_MASK
        # UUID のハッシュ値の下位 27bit とローカルスタイル ID の 5bit を組み合わせる
        combined_id = (uuid_hash << LOCAL_STYLE_ID_BITS) | local_style_id_masked
        # 32bit 符号付き整数として解釈するために、32bit 目が 1 の場合は正の値として扱う
        # 負の値にすると誤作動を引き起こす可能性があるため、符号ビットを反転させる
        if combined_id & SIGN_BIT:
            combined_id &= ~SIGN_BIT

        return StyleId(combined_id)

    @staticmethod
    def style_id_to_local_style_id(style_id: StyleId) -> int:
        """
        VOICEVOX ENGINE 互換のグローバルに一意な StyleId を AIVM マニフェスト内のローカルなスタイル ID に変換する

        Parameters
        ----------
        style_id : StyleId
            VOICEVOX ENGINE 互換のグローバルに一意なスタイル ID

        Returns
        -------
        local_style_id : int
            AIVM マニフェスト内のローカルなスタイル ID
        """

        # スタイル ID の下位 5 bit からローカルなスタイル ID を取り出す
        return style_id & 0x1F
