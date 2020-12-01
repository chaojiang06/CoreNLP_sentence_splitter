
# A sentence splitter wrapper for CoreNLP
## About
This wrapper returns the **untokenized** sentence splitting result from CoreNLP toolkit.

## Before starting
Please download [CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-latest.zip) and unzip everything to <code>stanford-corenlp-4.2.0</code> folder. If you want to work in Arabic, please download the [Arabic](http://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-arabic.jar) package and put it into <code>stanford-corenlp-4.2.0</code> folder.

The latest version for CoreNLP package and Arabic package can be found at their offical [website](https://stanfordnlp.github.io/CoreNLP/index.html).

## Usage
<code>python sentence_splitter_wrapper_for_CoreNLP_En.py</code>
<code>python sentence_splitter_wrapper_for_CoreNLP_Ar.py</code>

Each file contains an example sentence. The code will print out the splitting results.

## Update hisotry
This sentence splitter has gone through a few changes.
* Danqi Chen wrote the [original](https://github.com/facebookresearch/DrQA/blob/master/drqa/tokenizers/corenlp_tokenizer.py) python wrapper for the tokenization function in CoreNLP.
* Chao Jiang modified the code to make the sentence splitter produce split setnences with untokenized text.
* Wuwei Lan modified the code to make it works with the Arabic language.
