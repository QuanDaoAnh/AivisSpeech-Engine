# flake8: noqa

import asyncio
import glob
import hashlib
import re
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import BinaryIO, Final

import aivmlib
import httpx
from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    ModelArchitecture,
)
from fastapi import HTTPException
from semver.version import Version

from voicevox_engine import __version__
from voicevox_engine.logging import logger
from voicevox_engine.metas.Metas import (
    Speaker,
    SpeakerInfo,
    SpeakerStyle,
    SpeakerSupportedFeatures,
    StyleId,
    StyleInfo,
)
from voicevox_engine.metas.MetasStore import Character
from voicevox_engine.model import AivmInfo, LibrarySpeaker
from voicevox_engine.utility.user_agent_utility import generate_user_agent

__all__ = ["AivmManager"]


class AivmManager:
    """
    AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルと AIVM マニフェストを管理するクラス
    VOICEVOX ENGINE における MetasStore の役割を代替する (AivisSpeech Engine では MetasStore は無効化されている)
    AivisSpeech はインストールサイズを削減するため、AIVMX ファイルにのみ対応する
    ref: https://github.com/Aivis-Project/aivmlib#aivm-specification
    """

    # AivisSpeech でサポートされているマニフェストバージョン
    SUPPORTED_MANIFEST_VERSIONS: Final[list[str]] = ["1.0"]

    # AivisSpeech でサポートされている音声合成モデルのアーキテクチャ
    SUPPORTED_MODEL_ARCHITECTURES: Final[list[ModelArchitecture]] = [
        ModelArchitecture.StyleBertVITS2,
        ModelArchitecture.StyleBertVITS2JPExtra,
    ]

    # AivisHub API のベース URL
    AIVISHUB_API_BASE_URL: Final[str] = "https://api.aivis-project.com/v1"

    # デフォルトでインストールされる音声合成モデルの UUID
    DEFAULT_MODEL_UUIDS: Final[list[str]] = [
        "a59cb814-0083-4369-8542-f51a29e72af7",
    ]

    def __init__(self, installed_aivm_dir: Path):
        """
        AivmManager のコンストラクタ

        Parameters
        ----------
        installed_aivm_dir : Path
            AIVMX ファイルのインストール先ディレクトリ
        """

        self.installed_aivm_dir = installed_aivm_dir
        self.installed_aivm_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Models directory: {self.installed_aivm_dir}")

        # self.get_installed_aivm_infos() の実行結果のキャッシュ
        # すべてのインストール済み音声合成モデルの情報が保持される
        self._installed_aivm_infos: dict[str, AivmInfo] | None = None

        current_installed_aivm_infos = self.get_installed_aivm_infos()
        if len(current_installed_aivm_infos) == 0:
            logger.warning("No models are installed. Installing default models...")
            # デフォルトモデルをインストール
            for aivm_uuid in self.DEFAULT_MODEL_UUIDS:
                url = f"{self.AIVISHUB_API_BASE_URL}/aivm-models/{aivm_uuid}/download?model_type=AIVMX"
                logger.info(f"Installing default model from {url}...")
                self.install_aivm_from_url(url)
        else:
            logger.info("Installed models:")
            for aivm_info in current_installed_aivm_infos.values():
                logger.info(f"- {aivm_info.manifest.name} ({aivm_info.manifest.uuid})")

    def get_characters(self) -> list[Character]:
        """
        すべてのインストール済み音声合成モデル内の話者の一覧を Character 型で取得する (MetasStore 互換用)

        Returns
        -------
        characters : list[Character]
            インストール済み音声合成モデル内の話者の一覧
        """

        speakers = self.get_speakers()
        characters: list[Character] = []
        for speaker in speakers:
            character = Character(
                name=speaker.name,
                uuid=speaker.speaker_uuid,
                # AivisSpeech Engine では talk スタイルのみがサポートされる
                talk_styles=speaker.styles,
                # AivisSpeech Engine では歌唱音声合成はサポートされていない
                sing_styles=[],
                version=speaker.version,
                supported_features=speaker.supported_features,
            )
            characters.append(character)

        # 既に get_speakers() で話者名でソートされているのでそのまま返す
        return characters

    def get_speakers(self) -> list[Speaker]:
        """
        すべてのインストール済み音声合成モデル内の話者の一覧を取得する

        Returns
        -------
        speakers : list[Speaker]
            インストール済み音声合成モデル内の話者の一覧
        """

        aivm_infos = self.get_installed_aivm_infos()
        speakers: list[Speaker] = []
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                speakers.append(aivm_info_speaker.speaker)

        # 話者名でソートしてから返す
        return sorted(speakers, key=lambda x: x.name)

    def get_style_id_from_model_name(self, model_name: str) -> StyleId:
        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                if aivm_info_speaker.speaker.name == model_name:
                    return aivm_info.speakers[0].speaker.styles[0].id
        raise HTTPException(
            status_code=500,
            detail="model_name does not exist"
        )

    def get_speaker_info(self, speaker_uuid: str) -> SpeakerInfo:
        """
        インストール済み音声合成モデル内の話者の追加情報を取得する

        Parameters
        ----------
        speaker_uuid : str
            話者の UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        speaker_info : SpeakerInfo
            話者の追加情報
        """

        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                if aivm_info_speaker.speaker.speaker_uuid == speaker_uuid:
                    return aivm_info_speaker.speaker_info

        raise HTTPException(
            status_code=404,
            detail=f"話者 {speaker_uuid} はインストールされていません。",
        )

    def get_aivm_info(self, aivm_uuid: str) -> AivmInfo:
        """
        音声合成モデルの UUID から AIVMX ファイルの情報を取得する

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        aivm_info : AivmInfo
            AIVMX ファイルの情報
        """

        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            if str(aivm_info.manifest.uuid) == aivm_uuid:
                return aivm_info

        raise HTTPException(
            status_code=404,
            detail=f"音声合成モデル {aivm_uuid} はインストールされていません。",
        )

    def get_aivm_manifest_from_style_id(
        self, style_id: StyleId
    ) -> tuple[AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle]:
        """
        スタイル ID に対応する AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle を取得する

        Parameters
        ----------
        style_id : StyleId
            スタイル ID

        Returns
        -------
        aivm_manifest : AivmManifest
            AIVM マニフェスト
        aivm_manifest_speaker : AivmManifestSpeaker
            AIVM マニフェスト内の話者
        aivm_manifest_style : AivmManifestSpeakerStyle
            AIVM マニフェスト内のスタイル
        """

        # fmt: off
        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                for aivm_info_speaker_style in aivm_info_speaker.speaker.styles:
                    if aivm_info_speaker_style.id == style_id:
                        # ここでスタイル ID が示す音声合成モデルに対応する AivmManifest を特定
                        aivm_manifest = aivm_info.manifest
                        for aivm_manifest_speaker in aivm_manifest.speakers:
                            # ここでスタイル ID が示す話者に対応する AivmManifestSpeaker を特定
                            if str(aivm_manifest_speaker.uuid) == aivm_info_speaker.speaker.speaker_uuid:
                                for aivm_manifest_style in aivm_manifest_speaker.styles:
                                    # ここでスタイル ID が示すスタイルに対応する AivmManifestSpeakerStyle を特定
                                    local_style_id = self.style_id_to_local_style_id(style_id)
                                    if aivm_manifest_style.local_id == local_style_id:
                                        # すべて取得できたので値を返す
                                        return aivm_manifest, aivm_manifest_speaker, aivm_manifest_style

        raise HTTPException(
            status_code=404,
            detail=f"スタイル {style_id} は存在しません。",
        )

    def get_installed_aivm_infos(
        self, force: bool = False, wait_for_update_check: bool = False
    ) -> dict[str, AivmInfo]:
        """
        すべてのインストール済み音声合成モデルの情報を取得する

        Parameters
        ----------
        force : bool, default False
            強制的に再取得するかどうか
        wait_for_update_check : bool, default False
            AivisHub からの更新情報の取得を待機するかどうか

        Returns
        -------
        aivm_infos : dict[str, AivmInfo]
            インストール済み音声合成モデルの情報 (キー: 音声合成モデルの UUID, 値: AivmInfo)
        """

        # 既に取得済みかつ再取得が強制されていない場合は高速化のためキャッシュを返す
        if self._installed_aivm_infos is not None and not force:
            return self._installed_aivm_infos

        # AIVMX ファイルのインストール先ディレクトリ内に配置されている .aivmx ファイルのパスを取得
        aivm_file_paths = glob.glob(str(self.installed_aivm_dir / "*.aivmx")) + glob.glob(str(self.installed_aivm_dir / "*.aivm"))

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
                    else:
                        aivm_metadata = aivmlib.read_aivm_metadata(f)
                    aivm_manifest = aivm_metadata.manifest
            except aivmlib.AivmValidationError as ex:
                logger.warning(
                    f"{aivm_file_path}: Failed to read AIVM metadata. ({ex})"
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
            manifest_major, _ = map(int, manifest_version_parts)
            # サポート済みバージョンのメジャーバージョンを取得
            supported_major = int(self.SUPPORTED_MANIFEST_VERSIONS[0].split(".")[0])
            if manifest_major != supported_major:
                # メジャーバージョンが異なる場合はスキップ
                logger.warning(
                    f"{aivm_file_path}: AIVM manifest version {aivm_manifest.manifest_version} is not supported (different major version)."
                )
                continue
            elif aivm_manifest.manifest_version not in self.SUPPORTED_MANIFEST_VERSIONS:
                # 同じメジャーバージョンで、より新しいマイナーバージョンの場合は警告を出して続行
                logger.warning(
                    f"{aivm_file_path}: AIVM manifest version {aivm_manifest.manifest_version} is newer than supported versions. Trying to load anyway..."
                )

            # 音声合成モデルのアーキテクチャのバリデーション
            if aivm_manifest.model_architecture not in self.SUPPORTED_MODEL_ARCHITECTURES:  # fmt: skip
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
                speaker_icon = self.extract_base64_from_data_url(speaker_manifest.icon)

                # スタイルごとのメタデータを取得
                speaker_styles: list[SpeakerStyle] = []
                style_infos: list[StyleInfo] = []
                for style_manifest in speaker_manifest.styles:

                    # AIVM マニフェスト内の話者スタイル ID を VOICEVOX ENGINE 互換の StyleId に変換
                    style_id = self.local_style_id_to_style_id(style_manifest.local_id, speaker_uuid)  # fmt: skip

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
                        icon=self.extract_base64_from_data_url(style_manifest.icon) if style_manifest.icon else speaker_icon,
                        # 立ち絵を省略
                        ## VOICEVOX ENGINE 本家では portrait に立ち絵が入るが、AivisSpeech Engine では敢えてアイコン画像のみを設定する
                        portrait=None,
                        # ボイスサンプル
                        voice_samples=[
                            self.extract_base64_from_data_url(sample.audio)
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

        # 音声合成モデル名でソート
        sorted_aivm_infos = dict(sorted(aivm_infos.items(), key=lambda x: x[1].manifest.name))  # fmt: skip

        # キャッシュ更新前に、キャッシュに保持されている既存のロード状態を移行する
        if self._installed_aivm_infos is not None:
            for aivm_uuid, aivm_info in sorted_aivm_infos.items():
                if aivm_uuid in self._installed_aivm_infos:
                    aivm_info.is_loaded = self._installed_aivm_infos[aivm_uuid].is_loaded  # fmt: skip

        # 実行結果のキャッシュを更新
        self._installed_aivm_infos = sorted_aivm_infos

        # 非同期で AivisHub からの情報更新を開始
        # 音声合成エンジンの起動を遅延させないよう、別スレッドで非同期タスクを開始する
        try:
            if wait_for_update_check:
                # 更新情報の取得を待機する場合は同期的に実行
                asyncio.run(self.check_aivm_updates_from_hub())
            else:
                # 待機しない場合は別スレッドでバックグラウンド実行
                Thread(
                    target=asyncio.run, args=(self.check_aivm_updates_from_hub(),)
                ).start()
        except Exception as ex:
            # 非同期タスクの開始に失敗しても起動に影響を与えないよう、ログ出力のみ行う
            logger.warning(f"Failed to start async update task:", exc_info=ex)

        return self._installed_aivm_infos

    async def check_aivm_updates_from_hub(self) -> None:
        """
        AivisHub からすべてのインストール済み音声合成モデルのアップデート情報を取得し、非同期に更新する
        """

        async def fetch_latest_version(aivm_info: AivmInfo) -> None:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.AIVISHUB_API_BASE_URL}/aivm-models/{aivm_info.manifest.uuid}",
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

            except httpx.TimeoutException as ex:
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
        assert self._installed_aivm_infos is not None
        update_tasks = [
            fetch_latest_version(aivm_info)
            for aivm_info in self._installed_aivm_infos.values()
        ]

        # 全タスクを同時に実行
        await asyncio.gather(*update_tasks, return_exceptions=True)

        # 更新があった場合はログ出力
        update_available_models = [
            aivm_info
            for aivm_info in self._installed_aivm_infos.values()
            if aivm_info.is_update_available
        ]
        if update_available_models:
            logger.info("Update available models:")
            for aivm_info in update_available_models:
                logger.info(
                    f"- {aivm_info.manifest.name} ({aivm_info.manifest.uuid}) v{aivm_info.manifest.version} -> v{aivm_info.latest_version}"
                )

    def update_model_load_state(self, aivm_uuid: str, is_loaded: bool) -> None:
        """
        モデルのロード状態を更新する
        このメソッドは StyleBertVITS2TTSEngine 上でロード/アンロードされた際に呼び出される

        Parameters
        ----------
        aivm_uuid : str
            AIVM の UUID
        is_loaded : bool
            モデルがロードされているかどうか
        """

        if (
            self._installed_aivm_infos is not None
            and aivm_uuid in self._installed_aivm_infos
        ):
            self._installed_aivm_infos[aivm_uuid].is_loaded = is_loaded

    def install_aivm(self, file: BinaryIO) -> None:
        """
        AIVMX (Aivis Voice Model for ONNX) ファイル (`.aivmx`) をインストールする

        Parameters
        ----------
        file : BinaryIO
            AIVMX ファイルのバイナリ
        """

        # AIVMX ファイルからから AIVM メタデータを取得
        extension = "aivmx"
        try:
            aivm_metadata = aivmlib.read_aivmx_metadata(file)
            aivm_manifest = aivm_metadata.manifest
        except aivmlib.AivmValidationError as ex:
            try:
                aivm_metadata = aivmlib.read_aivm_metadata(file)
                aivm_manifest = aivm_metadata.manifest
                extension = "aivm"
            except aivmlib.AivmValidationError as ex:
                logger.error(f"AIVMX file is invalid:", exc_info=ex)
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された AIVMX ファイルの形式が正しくありません。({ex})",
                )

        # すでに同一 UUID のファイルがインストール済みの場合、同じファイルを更新する
        ## 手動で .aivmx ファイルをインストール先ディレクトリにコピーしていた (ファイル名が UUID と一致しない) 場合も更新できるよう、
        ## この場合のみ特別に更新先ファイル名を現在保存されているファイル名に変更する
        aivm_file_path = self.installed_aivm_dir / f"{aivm_manifest.uuid}.{extension}"
        if str(aivm_manifest.uuid) in self.get_installed_aivm_infos():
            logger.info(f"AIVM model {aivm_manifest.uuid} is already installed. Updating...")  # fmt: skip
            previous_aivm_info = self.get_installed_aivm_infos()[str(aivm_manifest.uuid)]  # fmt: skip
            # aivm_file_path を現在保存されているファイル名に変更
            aivm_file_path = previous_aivm_info.file_path

        # マニフェストバージョンのバリデーション
        if aivm_manifest.manifest_version not in self.SUPPORTED_MANIFEST_VERSIONS:  # fmt: skip
            logger.error(
                f"AIVM manifest version {aivm_manifest.manifest_version} is not supported."
            )
            raise HTTPException(
                status_code=422,
                detail=f"AIVM マニフェストバージョン {aivm_manifest.manifest_version} には対応していません。",
            )

        # 音声合成モデルのアーキテクチャのバリデーション
        if aivm_manifest.model_architecture not in self.SUPPORTED_MODEL_ARCHITECTURES:  # fmt: skip
            logger.error(
                f"AIVM model architecture {aivm_manifest.model_architecture} is not supported."
            )
            raise HTTPException(
                status_code=422,
                detail=f'モデルアーキテクチャ "{aivm_manifest.model_architecture}" には対応していません。',
            )

        # BinaryIO のシークをリセット
        # ここでリセットしないとファイルの内容を読み込めない
        file.seek(0)

        # AIVMX ファイルをインストール
        ## 通常は重複防止のため "(音声合成モデルの UUID).aivmx" のフォーマットのファイル名でインストールされるが、
        ## 手動で .aivmx ファイルをインストール先ディレクトリにコピーしても一通り動作するように考慮している
        logger.info(f"Installing AIVMX file to {aivm_file_path}...")
        try:
            with open(aivm_file_path, mode="wb") as f:
                f.write(file.read())
            logger.info(f"Installed AIVMX file to {aivm_file_path}.")
        except OSError as ex:
            logger.error(
                f"Failed to write AIVMX file to {aivm_file_path}:", exc_info=ex
            )
            error_message = str(ex).lower()
            if "no space" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。ストレージ容量が不足しています。({ex})"
            elif "permission denied" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。インストール先フォルダへのアクセス権限が不足しています。({ex})"
            elif "read-only" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。インストール先フォルダが読み取り専用権限になっています。({ex})"
            else:
                detail = f"AIVMX ファイルの書き込みに失敗しました。({ex})"
            raise HTTPException(
                status_code=500,
                detail=detail,
            )

        # すべてのインストール済み音声合成モデルの情報のキャッシュを再生成
        ## インストール完了後にエディタから送られる /aivm_models API へのリクエストで確実に更新情報も返せるように、
        ## 更新情報の取得が完了するのを待ってから戻る
        self.get_installed_aivm_infos(force=True, wait_for_update_check=True)

    def install_aivm_from_url(self, url: str) -> None:
        """
        指定された URL から AIVMX (Aivis Voice Model for ONNX) ファイル (`.aivmx`) をダウンロードしてインストールする

        Parameters
        ----------
        url : str
            AIVMX ファイルの URL
        """

        # AivisHub の音声合成モデル詳細ページの URL が渡された場合、特別に AivisHub API を使い AIVMX ファイルをダウンロードする
        if url.startswith("https://hub.aivis-project.com/aivm-models/"):
            # URL から AIVM の UUID を抽出
            uuid_match = re.search(
                r"/aivm-models/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                url.lower(),
            )
            if not uuid_match:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid AivisHub URL.",
                )
            # group(0) は一致した文字列全体なので、group(1) で UUID 部分のみを取得
            aivm_uuid = uuid_match.group(1)
            # AIVMX ダウンロード用の API の URL に置き換え
            url = f"{self.AIVISHUB_API_BASE_URL}/aivm-models/{aivm_uuid}/download?model_type=AIVMX"
            logger.info(
                f"Detected AivisHub model page URL. Using download API URL: {url}"
            )

        # URL から AIVMX ファイルをダウンロード
        try:
            logger.info(f"Downloading AIVMX file from {url}...")
            response = httpx.get(
                url,
                headers={"User-Agent": generate_user_agent()},
                # リダイレクトを追跡する
                follow_redirects=True,
            )
            response.raise_for_status()
            logger.info(f"Downloaded AIVMX file from {url}.")
        except httpx.HTTPError as ex:
            logger.error(f"Failed to download AIVMX file from {url}:", exc_info=ex)
            raise HTTPException(
                status_code=500,
                detail=f"AIVMX ファイルのダウンロードに失敗しました。({ex})",
            )

        # ダウンロードした AIVMX ファイルをインストール
        self.install_aivm(BytesIO(response.content))

    def update_aivm(self, aivm_uuid: str) -> None:
        """
        AivisHub から指定された音声合成モデルの一番新しいバージョンをダウンロードし、
        インストール済みの音声合成モデルへ上書き更新する

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)
        """

        # 対象の音声合成モデルがインストール済みかを確認
        installed_aivm_infos = self.get_installed_aivm_infos()
        if aivm_uuid not in installed_aivm_infos.keys():
            raise HTTPException(
                status_code=404,
                detail=f"音声合成モデル {aivm_uuid} はインストールされていません。",
            )

        # アップデートが利用可能かを確認
        aivm_info = installed_aivm_infos[aivm_uuid]
        if not aivm_info.is_update_available:
            raise HTTPException(
                status_code=422,
                detail=f"音声合成モデル {aivm_uuid} にアップデートはありません。",
            )

        # AivisHub からアップデートをダウンロードしてインストール
        logger.info(
            f"Updating AIVM model {aivm_uuid} to version {aivm_info.latest_version}..."
        )
        download_url = f"{self.AIVISHUB_API_BASE_URL}/aivm-models/{aivm_uuid}/download?model_type=AIVMX"
        self.install_aivm_from_url(download_url)
        logger.info(
            f"Updated AIVM model {aivm_uuid} to version {aivm_info.latest_version}."
        )

    def uninstall_aivm(self, aivm_uuid: str) -> None:
        """
        インストール済み音声合成モデルをアンインストールする

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)
        """

        # 対象の音声合成モデルがインストール済みかを確認
        installed_aivm_infos = self.get_installed_aivm_infos()
        if aivm_uuid not in installed_aivm_infos.keys():
            raise HTTPException(
                status_code=404,
                detail=f"音声合成モデル {aivm_uuid} はインストールされていません。",
            )

        # インストール済みの音声合成モデルの数を確認
        if len(installed_aivm_infos) <= 1:
            logger.error("AivisSpeech Engine must have at least one installed model.")
            raise HTTPException(
                status_code=400,
                detail="AivisSpeech Engine には必ず 1 つ以上の音声合成モデルがインストールされている必要があります。",
            )

        # AIVMX ファイルをアンインストール
        ## AIVMX ファイルのファイル名は必ずしも "(音声合成モデルの UUID).aivmx" になるとは限らないため、
        ## AivmInfo 内に格納されているファイルパスを使って削除する
        ## 万が一 AIVMX ファイルが存在しない場合は無視する
        logger.info(f"Uninstalling AIVMX file from {installed_aivm_infos[aivm_uuid].file_path}...")  # fmt: skip
        installed_aivm_infos[aivm_uuid].file_path.unlink(missing_ok=True)
        logger.info(f"Uninstalled AIVMX file from {installed_aivm_infos[aivm_uuid].file_path}.")  # fmt: skip

        # すべてのインストール済み音声合成モデルの情報のキャッシュを再生成
        ## インストール完了後にエディタから送られる /aivm_models API へのリクエストで確実に更新情報も返せるように、
        ## 更新情報の取得が完了するのを待ってから戻る
        self.get_installed_aivm_infos(force=True, wait_for_update_check=True)

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
