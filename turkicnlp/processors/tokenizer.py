"""
Tokenizers for Latin and Cyrillic script Turkic languages.

Provides :class:`RegexTokenizer` (rule-based, handles ~80% of cases)
and :class:`NeuralTokenizer` (BiLSTM character-level, for edge cases).
"""

from __future__ import annotations

import re
from typing import Optional

from turkicnlp.models.document import Document, Sentence, Token, Word
from turkicnlp.processors.base import Processor


class RegexTokenizer(Processor):
    """Rule-based tokenizer for Latin/Cyrillic Turkic languages.

    Most Turkic languages are space-delimited with predictable punctuation.
    This handles the majority of cases.
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []

    SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
    TOKEN_SPLIT = re.compile(r"(\S+)")
    PUNCT_DETACH = re.compile(r'^(.*?)([.,:;!?"\'\)\]\}]+)$')

    def __init__(
        self,
        lang: str,
        script: Optional[object] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang, script, config)
        self._abbreviations: set[str] = set()

    def load(self, model_path: Optional[str] = None) -> None:
        """Load optional abbreviation lists to avoid false sentence splits."""
        self._abbreviations = set()
        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Split text into sentences and tokens."""
        text = doc.text

        raw_sents = self.SENT_SPLIT.split(text)
        raw_sents = [s.strip() for s in raw_sents if s.strip()]

        char_offset = 0
        for sent_text in raw_sents:
            sentence = Sentence(text=sent_text)

            word_id = 1
            for match in self.TOKEN_SPLIT.finditer(sent_text):
                raw_token = match.group()
                token_start = char_offset + match.start()

                punct_match = self.PUNCT_DETACH.match(raw_token)
                if punct_match and punct_match.group(1):
                    word_text = punct_match.group(1)
                    word = Word(
                        id=word_id,
                        text=word_text,
                        start_char=token_start,
                        end_char=token_start + len(word_text),
                    )
                    token = Token(
                        id=(word_id,),
                        text=word_text,
                        words=[word],
                        start_char=token_start,
                        end_char=token_start + len(word_text),
                    )
                    sentence.tokens.append(token)
                    sentence.words.append(word)
                    word_id += 1

                    punct_text = punct_match.group(2)
                    punct_start = token_start + len(word_text)
                    for ch in punct_text:
                        pw = Word(
                            id=word_id,
                            text=ch,
                            upos="PUNCT",
                            lemma=ch,
                            start_char=punct_start,
                            end_char=punct_start + 1,
                        )
                        pt = Token(
                            id=(word_id,),
                            text=ch,
                            words=[pw],
                            start_char=punct_start,
                            end_char=punct_start + 1,
                        )
                        sentence.tokens.append(pt)
                        sentence.words.append(pw)
                        word_id += 1
                        punct_start += 1
                else:
                    word = Word(
                        id=word_id,
                        text=raw_token,
                        start_char=token_start,
                        end_char=token_start + len(raw_token),
                    )
                    token = Token(
                        id=(word_id,),
                        text=raw_token,
                        words=[word],
                        start_char=token_start,
                        end_char=token_start + len(raw_token),
                    )
                    sentence.tokens.append(token)
                    sentence.words.append(word)
                    word_id += 1

            doc.sentences.append(sentence)
            char_offset += len(sent_text) + 1

        doc._processor_log.append("tokenize:regex")
        return doc


class NeuralTokenizer(Processor):
    """Neural tokenizer based on a character-level BiLSTM.

    For languages where rule-based splitting is insufficient.
    Architecture follows Stanza's tokenizer.
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []

    def load(self, model_path: str) -> None:
        """Load trained character-level BiLSTM tokenizer model."""
        raise NotImplementedError("NeuralTokenizer.load not yet implemented.")

    def process(self, doc: Document) -> Document:
        """Neural sentence splitting and tokenization."""
        raise NotImplementedError("NeuralTokenizer.process not yet implemented.")
