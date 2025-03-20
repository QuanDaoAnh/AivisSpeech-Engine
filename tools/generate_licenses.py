import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Literal, assert_never


class LicenseError(Exception):
    # License違反があった場合、このエラーを出します。
    pass


class License:
    def __init__(
        self,
        package_name: str,
        package_version: str | None,
        license_name: str | None,
        license_text: str,
        license_text_type: Literal["raw", "local_address", "remote_address"],
    ):
        self.package_name = package_name
        self.package_version = package_version
        self.license_name = license_name

        if license_text_type == "raw":
            self.license_text = license_text
        elif license_text_type == "local_address":
            # ライセンステキストをローカルのライセンスファイルから抽出する
            self.license_text = Path(license_text).read_text(encoding="utf8")
        elif license_text_type == "remote_address":
            self.license_text = get_license_text(license_text)
        else:
            assert_never("型で保護され実行されないはずのパスが実行されました")


def get_license_text(text_url: str) -> str:
    """URL が指すテキストを取得する。"""
    with urllib.request.urlopen(text_url) as res:
        # NOTE: `urlopen` 返り値の型が貧弱なため型チェックを無視する
        return res.read().decode()  # type: ignore


def generate_licenses() -> list[License]:
    licenses: list[License] = []

    # pip
    try:
        pip_licenses_output = subprocess.run(
            [
                sys.executable,
                "-m",
                "piplicenses",
                "--from=mixed",
                "--format=json",
                "--with-urls",
                "--with-license-file",
                "--no-license-path",
            ],
            capture_output=True,
            check=True,
        ).stdout.decode()
    except subprocess.CalledProcessError as err:
        raise Exception(
            f"command output:\n{err.stderr and err.stderr.decode()}"
        ) from err

    licenses_json = json.loads(pip_licenses_output)
    for license_json in licenses_json:
        # ライセンス文を pip 外で取得されたもので上書きする
        package_name: str = license_json["Name"].lower()
        if license_json["LicenseText"] == "UNKNOWN":
            if package_name == "core" and license_json["Version"] == "0.0.0":
                continue
            # NVIDIA ランタイムはライセンス生成をスキップ
            if package_name.startswith("nvidia-"):
                continue
            elif package_name == "e2k":
                text_url = "https://raw.githubusercontent.com/Patchethium/e2k/master/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "future":
                text_url = "https://raw.githubusercontent.com/PythonCharmers/python-future/master/LICENSE.txt"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "license-expression":
                text_url = "https://raw.githubusercontent.com/aboutcode-org/license-expression/main/license-expression.ABOUT"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "packageurl-python":
                text_url = "https://raw.githubusercontent.com/package-url/packageurl-python/main/mit.LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "pefile":
                text_url = "https://raw.githubusercontent.com/erocarrera/pefile/master/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "pip-requirements-parser":
                text_url = "https://raw.githubusercontent.com/aboutcode-org/pip-requirements-parser/main/apache-2.0.LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "python-multipart":
                text_url = "https://raw.githubusercontent.com/andrew-d/python-multipart/master/LICENSE.txt"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "romkan":
                text_url = "https://raw.githubusercontent.com/soimort/python-romkan/master/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "distlib":
                text_url = "https://bitbucket.org/pypa/distlib/raw/7d93712134b28401407da27382f2b6236c87623a/LICENSE.txt"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "jsonschema":
                text_url = "https://raw.githubusercontent.com/python-jsonschema/jsonschema/dbc398245a583cb2366795dc529ae042d10c1577/COPYING"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "lockfile":
                text_url = "https://opendev.org/openstack/pylockfile/raw/tag/0.12.2/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "platformdirs":
                text_url = "https://raw.githubusercontent.com/platformdirs/platformdirs/aa671aaa97913c7b948567f4d9c77d4f98bfa134/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "webencodings":
                text_url = "https://raw.githubusercontent.com/gsnedders/python-webencodings/fa2cb5d75ab41e63ace691bc0825d3432ba7d694/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "antlr4-python3-runtime":
                text_url = "https://raw.githubusercontent.com/antlr/antlr4/v4.11.1/LICENSE.txt"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "flatbuffers":
                text_url = "https://raw.githubusercontent.com/google/flatbuffers/v24.3.25/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "gputil":
                text_url = "https://raw.githubusercontent.com/anderskm/gputil/refs/heads/master/LICENSE.txt"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "gradio_client":
                text_url = "https://raw.githubusercontent.com/gradio-app/gradio/v3.41.0/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "jieba":
                text_url = "https://raw.githubusercontent.com/fxsjy/jieba/v0.42.1/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "loguru":
                text_url = "https://raw.githubusercontent.com/Delgan/loguru/0.7.3/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "primepy":
                text_url = "https://raw.githubusercontent.com/janaindrajit/primePy/9c98276fee5211e8761dfc03c9a1e02127e09e4a/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "pyproject_hooks":
                text_url = "https://raw.githubusercontent.com/pypa/pyproject-hooks/v1.1.0/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "pywin32":
                continue  # ライセンスファイルがリポジトリにない
            elif package_name == "safetensors":
                text_url = "https://raw.githubusercontent.com/huggingface/safetensors/v0.4.3/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "sentencepiece":
                text_url = "https://raw.githubusercontent.com/google/sentencepiece/v0.2.0/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "sudachipy":
                text_url = "https://raw.githubusercontent.com/WorksApplications/sudachi.rs/v0.6.8/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "tokenizers":
                text_url = "https://raw.githubusercontent.com/huggingface/tokenizers/v0.19.1/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "triton":
                text_url = "https://raw.githubusercontent.com/triton-lang/triton/v2.1.0/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "types-pyyaml":
                text_url = "https://raw.githubusercontent.com/python/typeshed/57f3dcac8dbed008479b251512975901a0206deb/LICENSE"  # noqa: B950
                license_json["LicenseText"] = get_license_text(text_url)
            elif package_name == "wmi":
                continue  # ライセンスファイルがリポジトリにない
            else:
                # ライセンスがpypiに無い
                raise Exception(f"No License info provided for {package_name}")
        # soxr
        if package_name == "soxr":
            text_url = "https://raw.githubusercontent.com/dofuuz/python-soxr/v0.3.6/LICENSE.txt"  # noqa: B950
            license_json["LicenseText"] = get_license_text(text_url)

        license = License(
            package_name=license_json["Name"],
            package_version=license_json["Version"],
            license_name=license_json["License"],
            license_text=license_json["LicenseText"],
            license_text_type="raw",
        )

        licenses.append(license)

    python_version = "3.11.9"

    licenses = [
        License(
            package_name="VOICEVOX ENGINE",
            package_version=None,
            license_name="LGPL license",
            license_text="https://raw.githubusercontent.com/VOICEVOX/voicevox_engine/master/LGPL_LICENSE",
            license_text_type="remote_address",
        ),
        # https://sourceforge.net/projects/open-jtalk/files/Open%20JTalk/open_jtalk-1.11/
        License(
            package_name="Open JTalk",
            package_version="1.11",
            license_name="Modified BSD license",
            license_text="tools/licenses/open_jtalk/COPYING",
            license_text_type="local_address",
        ),
        License(
            package_name="MeCab",
            package_version=None,
            license_name="Modified BSD license",
            license_text="tools/licenses/open_jtalk/mecab/COPYING",
            license_text_type="local_address",
        ),
        License(
            package_name="NAIST Japanese Dictionary",
            package_version=None,
            license_name="Modified BSD license",
            license_text="tools/licenses/open_jtalk/mecab-naist-jdic/COPYING",
            license_text_type="local_address",
        ),
        License(
            package_name="PyTorch",
            package_version="2.3.1",
            license_name="BSD-style license",
            license_text="https://raw.githubusercontent.com/pytorch/pytorch/master/LICENSE",
            license_text_type="remote_address",
        ),
        License(
            package_name="Python",
            package_version=python_version,
            license_name="Python Software Foundation License",
            license_text=f"https://raw.githubusercontent.com/python/cpython/v{python_version}/LICENSE",
            license_text_type="remote_address",
        ),
    ] + licenses  # 前方に追加

    return licenses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    output_path = args.output_path

    licenses = generate_licenses()

    # dump
    out = Path(output_path).open("w") if output_path else sys.stdout
    json.dump(
        [
            {
                "name": license.package_name,
                "version": license.package_version,
                "license": license.license_name,
                "text": license.license_text,
            }
            for license in licenses
        ],
        out,
    )
