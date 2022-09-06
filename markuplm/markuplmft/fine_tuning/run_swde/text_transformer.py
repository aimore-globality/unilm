import string
from typing import Optional
import re2 as re
from html import unescape
import unidecode


class TextTransformer:
    def __init__(self):
        self.transform_sequence = [
            self.decode,
            self.lower,
            self.decode_unicode,
            self.remove_common_words,
            self.replace_symbols,
            self.remove_symbols,
            self.normalize_any_space,
        ]
        print("Text Transformation Sequence:")
        for transformation in self.transform_sequence:
            print(f"{transformation.__qualname__}")

    @staticmethod
    def decode(input_text: str) -> str:
        return unidecode.unidecode(input_text)

    @staticmethod
    def lower(input_text: str) -> str:
        return input_text.lower()

    @staticmethod
    def decode_unicode(input_text: str) -> str:
        return unescape(input_text)

    @staticmethod
    def replace_symbols(input_text: str, replacing_symbol=" ") -> str:
        all_symbols = set(string.punctuation) - {"&"}
        for symbol in all_symbols:
            input_text = input_text.replace(symbol, replacing_symbol)
        return input_text

    def remove_symbols(self, input_text: str) -> str:
        return self.replace_symbols(input_text, "")

    @staticmethod
    def remove_common_words(input_text: str) -> str:
        input_text = f" {input_text.strip()} "
        common_words = ["the", "enterprises", "group", "plc", "company", "limited"]
        for common_word in common_words:
            input_text = input_text.replace(f" {common_word} ", " ")
        return input_text

    @staticmethod
    def normalize_any_space(input_text: str) -> str:
        input_text = re.sub("\s+", " ", input_text)
        input_text = f" {input_text.strip()} "
        return input_text

    def transform(self, text: str) -> Optional[str]:
        if text is not None:
            for transformation in self.transform_sequence:
                text = transformation(text)
            return text
