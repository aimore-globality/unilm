import string
import re2 as re
from html import unescape
import unidecode

class TextTransformer:
    def __init__(self, transformations=["decode", "lower", "decode_ampersand", "replace_symbols", "remove_symbols", "remove_common_words", "normalize_any_space"]):
        self.transformations = transformations
        all_transformations = dict(
            decode=self.decode,
            lower=self.lower,
            decode_ampersand=self.decode_ampersand,
            replace_symbols=self.replace_symbols,
            remove_symbols=self.remove_symbols,
            remove_common_words=self.remove_common_words,
            normalize_any_space=self.normalize_any_space,
        )
        self.transform_sequence = [all_transformations.get(transformation) for transformation in transformations]
        print(self.transform_sequence)
        
    @staticmethod
    def decode(input_text:str):
        return unidecode.unidecode(input_text)

    @staticmethod
    def lower(input_text:str):
        return input_text.lower()
    
    @staticmethod
    def decode_ampersand(input_text:str):
        return unescape(input_text)
        # input_text = input_text.replace("amp&;", '&')
        # return input_text.replace('amp&', '&')

    @staticmethod
    def replace_symbols(input_text:str, symbol_to_replace=' '):
        all_symbols = set(string.punctuation) - {'&'}
        for symbol in all_symbols:
            input_text = input_text.replace(symbol, symbol_to_replace)
        return input_text

    @staticmethod
    def remove_symbols(input_text:str): 
        all_symbols = set(string.punctuation) - {'&'}
        for symbol in all_symbols:
            input_text = input_text.replace(symbol, '')
        return input_text

    @staticmethod
    def remove_common_words(input_text:str):
        input_text = f" {input_text.strip()} "
        common_words = ["the", "enterprises", "group", "plc", "company", "limited"]
        for common_word in common_words:
            input_text = input_text.replace(f" {common_word} ", " ")
        return input_text

    @staticmethod
    def normalize_any_space(input_text:str):
        input_text = re.sub('\s+', ' ', input_text)
        input_text = f" {input_text.strip()} "
        return input_text

    def transform(self, text):
        if text is not None: 
            for transformation in self.transform_sequence:
                text = transformation(text)
            return text