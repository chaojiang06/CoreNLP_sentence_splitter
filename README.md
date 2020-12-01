
# A sentence splitter wrapper for CoreNLP
## About
This wrapper returns the **untokenized** sentence splitting result from CoreNLP toolkit.

## Before starting
Please download [CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-latest.zip) and unzip it. 
After done, please replace the <code>CORENLP_PATH</code>  with your path to the CoreNLP folder. And keep the <code>/*</code> at the end of the path. 

## Usage
<code>python sentence_splitter_wrapper_for_CoreNLP.py</code>

It should return an example output: <code>['Tom graduated from  No. 1 Middle School.', 'He loves reading.']</code>

## Work in Arabic
If you want to work in the Arabic language, please download the [Arabic model](http://nlp.stanford.edu/software/stanford-arabic-corenlp-models-current.jar) and put it in the CoreNLP folder above. Models in different languages are listed [here](https://github.com/stanfordnlp/CoreNLP) and [here](https://stanfordnlp.github.io/CoreNLP/human-languages.html#models).

You also need to adjust one line in the code to call the Arabic models in CoreNLP. Please see the <code>cmd</code> variable in <code>_launch</code> function in <code>CoreNLPTokenizer</code> class.

## Update hisotry
This sentence splitter has gone through a few changes.
* Danqi Chen wrote the [original](https://github.com/facebookresearch/DrQA/blob/master/drqa/tokenizers/corenlp_tokenizer.py) python wrapper for the tokenization function in CoreNLP.
* Chao Jiang modified the code to make the sentence splitter produce split setnences with untokenized text.
* Wuwei Lan modified the code to make it works with the Arabic language.
