# Deep contextualized word representations
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

ELMo is one of the first few papers that established the power of transfer learning in NLP. It was a huge deal when it was introduced, creating a new state of the art in almost all NLP tasks. As compared to traditional word embeddings such as word2vec and Glove, ELMo embeddings model context pretty well giving different embeddings to a word based on the sense it is used in a sentence.

## Quick Overview
This paper proposes a type of word representation that:
* models characteristics of word use i.e. syntax and semantics
* is context-dependent
* is deep i.e. a function of all internal layers of a biLM
* is character-based as initial context independent embedding is got from character convolutions as detailed below

## Summary
* Deep contexualized word representations differ from traditional word representations such as word2vec and Glove in that they are context-dependent and the representation for each word is a function of an entire sentence in which it appears.
* The representations are obtained from a biLM trained on a large text corpus with a language model objective. Some weights are shared between both the directions instead of being completely independent.
* A task specific weighted sum of each hidden state is computed. The task specific weight is learned dependent on the task the embedding is used for. 
* The ELMo embeddings can be added to any supervised NLP task by concatenating ELMo embedding with context-independent word embeddings. The ELMo embeddings can additionally also be concatenated at the output also. Adding dropout to ELMo is found to be beneficial.
* As character based convolutional filters are used, ELMo can handle out of the vocabulary words.
* Finetuning biLM on domain specific data leads to good improvement in downstream task.

## Pre-trained bidirectional language model Architecture
* Language Model: 
    * biLSTM with 2 layers, 4096 units and 512 dimension projections 
    * biLSTM has a residual connection from first to second layer

* Context-independent word embeddings:
    * 2048 character n-gram convolutional filters
    * This is followed by two highway layers and a linear projection down to a 512 representation

## Results
* ELMo embeddings can be easily added to existing models.
* They significantly improve the state of the art across six challenging NLP problems, including Question Answering, Textual Entailment, Coreference resolution, Semantic Role Labeling, Named entity extraction and Sentiment analysis with relative error reductions from 6-20% over strong base models.
* First layer of biLM seems to capture semantic information while top layer seems to capture the context.
* ELMo models use smaller training sets more efficiently than models without ELMo.

## Resources
* [Paper](https://arxiv.org/pdf/1802.05365.pdf)
* [Video](https://vimeo.com/277672840#t=183s)

## Implementation
* Tensorflow implementation supporting both training biLMs and pre-trained models can be found [here.](https://github.com/allenai/bilm-tf)
* For easier way to get ELMo embeddings check [this](https://tfhub.dev/google/elmo/3) out.
* See [ELMo](https://allennlp.org/elmo) page for more details.
