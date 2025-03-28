"""ユーザー辞書機能を提供する API Router"""

from typing import Annotated, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from pydantic import ValidationError
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.user_dict.constants import (
    USER_DICT_MAX_PRIORITY,
    USER_DICT_MIN_PRIORITY,
    WordProperty,
    WordTypes,
)
from voicevox_engine.user_dict.model import (
    UserDictInputError,
    UserDictWord,
    UserDictWordForCompat,
)
from voicevox_engine.user_dict.user_dict_manager import UserDictionary

from ..dependencies import VerifyMutabilityAllowed


def generate_user_dict_router(
    user_dict: UserDictionary, verify_mutability: VerifyMutabilityAllowed
) -> APIRouter:
    """ユーザー辞書 API Router を生成する"""
    router = APIRouter(tags=["ユーザー辞書"])

    @router.get(
        "/user_dict",
        summary="ユーザー辞書に登録されている単語の一覧を取得する",
        response_description="単語の UUID とその詳細",
    )
    def get_user_dict_words(
        enable_compound_accent: Annotated[
            bool,
            Query(
                description="複数のアクセント句を持つ単語の扱いを指定します。false の場合は API 互換性のため、最初のアクセント句の情報のみを返します。"
            ),
        ] = False,
    ) -> dict[str, UserDictWord | UserDictWordForCompat]:
        """
        ユーザー辞書に登録されている単語の一覧を返します。
        単語の表層形 (surface) は正規化済みの物を返します。
        """
        try:
            all_words = user_dict.get_all_words()
            if enable_compound_accent is True:
                # enable_compound_accent=True の時は UserDictWord をそのまま返す
                return cast(dict[str, UserDictWord | UserDictWordForCompat], all_words)
            else:
                # enable_compound_accent=False の時は UserDictWordForCompat に変換してから返す
                return {
                    word_uuid: UserDictWordForCompat.from_user_dict_word(user_dict_word)
                    for word_uuid, user_dict_word in all_words.items()
                }
        except UserDictInputError as err:
            raise HTTPException(status_code=422, detail=str(err))
        except Exception:
            raise HTTPException(
                status_code=500, detail="辞書の読み込みに失敗しました。"
            )

    # TODO: CsvSafeStrを使う
    @router.post(
        "/user_dict_word",
        dependencies=[Depends(verify_mutability)],
        summary="ユーザー辞書に単語を追加する",
        response_description="追加した単語の UUID",
    )
    def add_user_dict_word(
        surface: Annotated[list[str], Query(description="単語の表層形")],
        pronunciation: Annotated[
            list[str], Query(description="単語の発音（カタカナ）")
        ],
        accent_type: Annotated[
            list[int], Query(description="アクセント型（音が下がる場所を指す）")
        ],
        word_type: Annotated[
            WordTypes | SkipJsonSchema[None],
            Query(
                description="PROPER_NOUN（固有名詞）、LOCATION_NAME（地名）、ORGANIZATION_NAME（組織・施設名）、PERSON_NAME（人名）、PERSON_FAMILY_NAME（姓）、PERSON_GIVEN_NAME（名）、COMMON_NOUN（普通名詞）、VERB（動詞）、ADJECTIVE（形容詞）、SUFFIX（語尾）のいずれか"
            ),
        ] = None,
        priority: Annotated[
            int | SkipJsonSchema[None],
            Query(
                ge=USER_DICT_MIN_PRIORITY,
                le=USER_DICT_MAX_PRIORITY,
                description="単語の優先度（0から10までの整数）。数字が大きいほど優先度が高くなる。1から9までの値を指定することを推奨。",
                # "SkipJsonSchema[None]"の副作用でスキーマーが欠落する問題に対するワークアラウンド
                json_schema_extra={
                    "le": None,
                    "ge": None,
                    "maximum": USER_DICT_MAX_PRIORITY,
                    "minimum": USER_DICT_MIN_PRIORITY,
                },
            ),
        ] = None,
    ) -> str:
        """
        ユーザー辞書に単語を追加します。
        """
        try:
            word_uuid = user_dict.add_word(
                WordProperty(
                    surface=surface,
                    pronunciation=pronunciation,
                    accent_type=accent_type,
                    word_type=word_type,
                    priority=priority,
                )
            )
            return word_uuid
        except ValidationError as ex:
            raise HTTPException(
                status_code=422, detail="パラメータに誤りがあります。\n" + str(ex)
            )
        except UserDictInputError as err:
            raise HTTPException(status_code=422, detail=str(err))
        except Exception:
            raise HTTPException(
                status_code=500, detail="ユーザー辞書への追加に失敗しました。"
            )

    @router.put(
        "/user_dict_word/{word_uuid}",
        status_code=204,
        dependencies=[Depends(verify_mutability)],
        summary="ユーザー辞書に登録されている単語を更新する",
    )
    def rewrite_user_dict_word(
        surface: Annotated[list[str], Query(description="単語の表層形")],
        pronunciation: Annotated[
            list[str], Query(description="単語の発音（カタカナ）")
        ],
        accent_type: Annotated[
            list[int], Query(description="アクセント型（音が下がる場所を指す）")
        ],
        word_uuid: Annotated[str, Path(description="更新する単語の UUID")],
        word_type: Annotated[
            WordTypes | SkipJsonSchema[None],
            Query(
                description="PROPER_NOUN（固有名詞）、LOCATION_NAME（地名）、ORGANIZATION_NAME（組織・施設名）、PERSON_NAME（人名）、PERSON_FAMILY_NAME（姓）、PERSON_GIVEN_NAME（名）、COMMON_NOUN（普通名詞）、VERB（動詞）、ADJECTIVE（形容詞）、SUFFIX（語尾）のいずれか"
            ),
        ] = None,
        priority: Annotated[
            int | SkipJsonSchema[None],
            Query(
                ge=USER_DICT_MIN_PRIORITY,
                le=USER_DICT_MAX_PRIORITY,
                description="単語の優先度（0から10までの整数）。数字が大きいほど優先度が高くなる。1から9までの値を指定することを推奨。",
                # "SkipJsonSchema[None]"の副作用でスキーマーが欠落する問題に対するワークアラウンド
                json_schema_extra={
                    "le": None,
                    "ge": None,
                    "maximum": USER_DICT_MAX_PRIORITY,
                    "minimum": USER_DICT_MIN_PRIORITY,
                },
            ),
        ] = None,
    ) -> None:
        """
        ユーザー辞書に登録されている単語を更新します。
        """
        try:
            user_dict.update_word(
                word_uuid,
                WordProperty(
                    surface=surface,
                    pronunciation=pronunciation,
                    accent_type=accent_type,
                    word_type=word_type,
                    priority=priority,
                ),
            )
        except ValidationError as ex:
            raise HTTPException(
                status_code=422, detail="パラメータに誤りがあります。\n" + str(ex)
            )
        except UserDictInputError as err:
            raise HTTPException(status_code=422, detail=str(err))
        except Exception:
            raise HTTPException(
                status_code=500, detail="ユーザー辞書の更新に失敗しました。"
            )

    @router.delete(
        "/user_dict_word/{word_uuid}",
        status_code=204,
        dependencies=[Depends(verify_mutability)],
        summary="ユーザー辞書に登録されている単語を削除する",
    )
    def delete_user_dict_word(
        word_uuid: Annotated[str, Path(description="削除する単語の UUID")],
    ) -> None:
        """
        ユーザー辞書に登録されている単語を削除します。
        """
        try:
            user_dict.delete_word(word_uuid=word_uuid)
        except UserDictInputError as err:
            raise HTTPException(status_code=422, detail=str(err))
        except Exception:
            raise HTTPException(
                status_code=500, detail="ユーザー辞書の更新に失敗しました。"
            )

    @router.post(
        "/import_user_dict",
        status_code=204,
        dependencies=[Depends(verify_mutability)],
        summary="他のユーザー辞書をインポートする",
    )
    def import_user_dict_words(
        import_dict_data: Annotated[
            dict[str, UserDictWord | UserDictWordForCompat],
            Body(description="インポートするユーザー辞書のデータ"),
        ],
        override: Annotated[
            bool, Query(description="重複したエントリがあった場合、上書きするかどうか")
        ],
    ) -> None:
        """
        他のユーザー辞書をインポートします。
        """
        try:
            converted_import_dict_data: dict[str, UserDictWord] = {}
            for word_uuid, user_dict_word in import_dict_data.items():
                # UserDictWordForCompat であれば UserDictWord に変換
                if isinstance(user_dict_word, UserDictWordForCompat):
                    converted_import_dict_data[word_uuid] = (
                        UserDictWord.from_user_dict_word_for_compat(user_dict_word)
                    )
                else:
                    converted_import_dict_data[word_uuid] = user_dict_word
            user_dict.import_dictionary(
                dict_data=converted_import_dict_data, override=override
            )
        except UserDictInputError as err:
            raise HTTPException(status_code=422, detail=str(err))
        except Exception:
            raise HTTPException(
                status_code=500, detail="ユーザー辞書のインポートに失敗しました。"
            )

    return router
