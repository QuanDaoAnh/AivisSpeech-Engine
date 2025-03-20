"""
/multi_synthesis API のテスト
"""

from test.e2e.single_api.utils import gen_mora

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion


def test_post_multi_synthesis_200(
    client: TestClient, snapshot: SnapshotAssertion
) -> None:
    queries = [
        {
            "accent_phrases": [
                {
                    "moras": [
                        gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                        gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                        gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                    ],
                    "accent": 1,
                    "pause_mora": None,
                    "is_interrogative": False,
                }
            ],
            "speedScale": 1.0,
            "pitchScale": 1.0,
            "intonationScale": 1.0,
            "volumeScale": 1.0,
            "prePhonemeLength": 0.1,
            "postPhonemeLength": 0.1,
            "pauseLength": None,
            "pauseLengthScale": 1.0,
            "outputSamplingRate": 44100,
            "outputStereo": False,
            "kana": "テスト",
        },
        {
            "accent_phrases": [
                {
                    "moras": [
                        gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                        gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                        gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                        gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                    ],
                    "accent": 1,
                    "pause_mora": None,
                    "is_interrogative": False,
                }
            ],
            "speedScale": 1.0,
            "pitchScale": 1.0,
            "intonationScale": 1.0,
            "volumeScale": 1.0,
            "prePhonemeLength": 0.2,
            "postPhonemeLength": 0.1,
            "pauseLength": None,
            "pauseLengthScale": 1.0,
            "outputSamplingRate": 44100,
            "outputStereo": False,
            "kana": "テストト",
        },
    ]
    response = client.post(
        "/multi_synthesis", params={"speaker": 888753760}, json=queries
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    # zip 内の全ての wav の波形がスナップショットと一致する
    # AivisSpeech Engine の音声合成は常にある程度のランダム性があるため、テストではハッシュ値の比較は行わない
    # zip_bytes = io.BytesIO(response.read())
    # with zipfile.ZipFile(zip_bytes, "r") as zip_file:
    #     wav_files = (zip_file.read(name) for name in zip_file.namelist())
    #     for wav in wav_files:
    #         assert snapshot == hash_wave_floats_from_wav_bytes(wav)
