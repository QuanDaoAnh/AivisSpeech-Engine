from voicevox_engine.user_dict.constants import PART_OF_SPEECH_DATA, WordTypes


def test_word_types() -> None:
    word_types = list(WordTypes)
    part_of_speeches = list(PART_OF_SPEECH_DATA.keys())
    assert sorted(word_types) == sorted(part_of_speeches)
