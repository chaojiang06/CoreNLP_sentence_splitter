#!/usr/bin/env python
# coding: utf-8

"""
A sentence splitter wrapper for CoreNLP. This wrapper returns the untokenized sentence splitting result from CoreNLP toolkit.
This sentence splitter has gone through a few changes.
(1) Danqi Chen wrote the original python wrapper for the tokenization function in CoreNLP, https://github.com/facebookresearch/DrQA/blob/master/drqa/tokenizers/corenlp_tokenizer.py
(2) Chao Jiang modified the code to make the sentence splitter produce splitted setnences with untokenized text.
(3) Wuwei Lan modified the code to make it works with Arabic language.

"""

import copy
import json
import pexpect

CORENLP_PATH = './stanford-corenlp-4.2.0/*'

"""Base tokenizer/tokens classes and utilities."""
class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None, output = None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}
        if output != None:
            self.output = output

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def ssplit(self):
        s_list = []
        original_sentence = self.untokenize()
        dict_a = self.output
        for i in dict_a['sentences']:
            start_offset = i['tokens'][0]['characterOffsetBegin']
            end_offset = i['tokens'][-1]['characterOffsetEnd']
#             s_list.append(original_sentence[start_offset:end_offset+1].strip())
            s_list.append(original_sentence[start_offset:end_offset].strip())

        return s_list

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """
    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()

"""Simple wrapper around the Stanford CoreNLP pipeline.

Serves commands to a java subprocess running the jar. Requires java 8.
"""
class CoreNLPTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        self.classpath = (CORENLP_PATH)
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self._launch()

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        # if you work on English, use this this command
        # cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
        #        'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
        #        annotators, '-tokenize.options', options,
        #        '-outputFormat', 'json', '-prettyPrint', 'false']
        
        # if you work on arabic, use this this command
        
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
               # 'edu.stanford.nlp.pipeline.StanfordCoreNLP','-annotators',
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-props', 'StanfordCoreNLP-arabic.properties','-annotators',
               annotators, '-tokenize.options', options, #'-tokenize.whitespace', 'true',
               '-outputFormat', 'json', '-prettyPrint', 'false']
        print(' '.join(cmd))

        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return Tokens(data, self.annotators)

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{\r\n  "sentences":')
        output = json.loads(output[start:].decode('utf-8'))
        
        data = []
        tokens = [t for s in output['sentences'] for t in s['tokens']]
#         print(output)
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1]['characterOffsetBegin']
            else:
                end_ws = tokens[i]['characterOffsetEnd']

            data.append((
                self._convert(tokens[i]['word']),
                text[start_ws: end_ws],
                (tokens[i]['characterOffsetBegin'],
                 tokens[i]['characterOffsetEnd']),
                tokens[i].get('pos', None),
                tokens[i].get('lemma', None),
                tokens[i].get('ner', None)
            ))
        return Tokens(data, self.annotators, output = output)

def corenlp_ssplitter(text):
    """
    A wrapper for CoreNLP sentence splitting function. 
    Input is a string, function will return a list of string.
    CoreNLP will quit if receives "q" or "Q". Therefore, returning [text] if text in ['q', 'Q']
    """
    
    
    if text not in ['q', 'Q']:
        return tok.tokenize(text).ssplit()
    else:
        return [text]

tok = CoreNLPTokenizer()
text = "تخرج توم من المدرسة الإعدادية رقم 1. يحب القراءة."
print(text)
print(corenlp_ssplitter(text))
assert " ".join(corenlp_ssplitter(text)) == text
