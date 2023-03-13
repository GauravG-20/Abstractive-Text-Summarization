# Abstractive Text summarization using Bi-directional Attentive sequence to sequence RNNs

## Abstract

Abstractive text summarization generates a shorter version of a given sentence while attempting to preserve its contextual meaning. In our approach we model the problem using an attentional encoder decoder which ensures that the decoder focuses on the appropriate input words at each step of our generation. Our model relies only on the pre-trained features and can easily be trained in an end to end manner on any large datasets. Our experiments show promising results on the Gigaword dataset using ROGUE as a benchmark for performance measure. 

## 1. Introduction

Generating a condensed version also known as a summary of a given text is known as text summarization. This is a very important task in the field of natural language modelling. Summarization systems can be classified into two categories. Extractive models generate summary by selecting some part of the original text and putting them together to generate a summary. Abstractive models on the other hand are not constrained by the phrases from the input text, they generate summaries from scratch.
We have proposed an approach of using a Bi-directional RNN on our encoding layer and attention mechanism on the decoding layer. The attention model to be used is proposed in Bahdanau et al.(2014)[1], which has produced state-of-the-art performance in machine translation (MT), which is also a natural language task.

The attention mechanism works in a way that at every time step the decoder takes a conditioning input which is the output of the encoder module, depending on the current state of the RNN, the encoder then computes the score over the input text. These scores can be considered as a soft alignment over the input text informing the decoder that which part of the sentence it should focus on. Both the decoder and encoder are jointly trained. The input vector consists of integer values that present a mapping of the text to vocabulary key challenge in the task of text summarization is that there is not a direct relation between the source and the target words in the generated summary. 

We have contributed in the following manner:

1.We preprocess the data vigorously and remove any form of irregularities.  
2. Developed our own word embeddings from conecptnet number batch, Concept net 5.5 et al. (2017) rather then glove to significantly improve performance as it provides a reduced bias of human like stereotypes.  
3.Experimenting with multiple model architectures are corpus to explore the best possible results and increased learning.   
4.Provide a comparative model analysis of different techniques used and their respective ROGUE scores.  

## 2. Methodology

### 2.1. Preparing the data

We loaded the raw data into a panda’s data frame, only the required columns of text and summary were kept and the rest were dropped. Rows containing no data or nan values were also removed. A list of contractions was downloaded from stackoverflow to expand them so they could provide a better context to our model. The data was converted to lower case and regular expressions were used to remove unwanted characters and stop words were also removed. The total size of the cleaned vocabulary was 132,884 words. 
The Conceptnet Number batch pre-trained word embeddings were used instead of glove as the provide better context and do not include human stereo-types which affect the affective training of our model. We used a threshold of 21 which means that if the word occurs in our corpus more then 21 times and is not in the conceptnet pre-trained embeddings then it should be initialized with random numbers. The words in our corpus that are missing from the conceptnet embeddings are 3203 which account for 2.41 percent of the total words. 
Total number of unique words are 132,884 and we will use 49 percent of them. We define the embedding dimension to be 300 to match the conceptnet vectors. Final length of word embedding matrix was 65,599. The 95 percentiles of our summary length were 9 and text length was 115 which gives us some good ideas of the distribution of length across our data. The data was then sorted to ensure that same size data would be sampled in one batch during training and would decrease the computation off head as less padding would be required.

### 2.2. Model details

The final model that we used after comparison of results was a bi-directional RNN with attention mechanism inspired by Bahdanau et al.(2014) to build a sequence to sequence encoder decoder model. Initially model placeholders are defined which include input data, targets, learning rate, keep probability or dropout rate, summary length, the max summary length and input text length. A <GO> token is appended to the start of each batch. In the encoding layer we have two LSTM cells the initialization used are same as those in Pan’s and Liu’s model[2]. The outputs are concatenated as it is a bi directional RNN. We have two sperate decoding functions one is the training decoder and the other is the inference decoder. Decoding cell is just two layered LSTM with dropout. The attention mechanism used is Bhadanau as it helps the model train faster. The Dynamic attention wrapper applies the attention mechanism to our decoding cell. The 	Dynamic attention wrapper state creates the initial state for our training and inference logits. Since it is a bidirectional RNN we can use both the forward and the backward state but we choose the forward state. Both the encoding and decoding sequences will use the same word embeddings that we created earlier. The decoder output layer is a dense layer with number of neurons equal to the vocabulary size and truncated normal initializer with mean 0 and standard deviation 0.1. Padding is done in a single batch to ensure uniformity, it is done using <PAD> token. After model training the text for which we need to generate a summary needs to be converted to integers as it is represented in that format in our embedding space. The tensors are loaded from the saved graphs and the input text is multiplied by batch size to match the input parameters of our model.

## 3. Experimentation

### 3.1 Dataset Description

The dataset that we used is the Gigaword dataset, it contains 568,411 text summary pairs. The main dataset is hosted on the stand ford SNAP server but the subset that we used is the food reviews dataset which is hosted on Kaggle.

### 3.2 Experiment details

We experimented quite a bit with different models ,we used GRU cells instead of LSTM cells and also tried unidirectional model structure but the best results that we were getting on a small subset of the data was of the bi directional RNN with LSTM cells and bhadanau attention.
The hyper-parameters that we used for our experimentation and final evaluation were as follows, the total epochs were set to 100, batch size was 64 .The batch size we couldn’t increase as the tensor size was already too large and more could not fit the 12 GB tesla k80M that we were using for training. The RNN hidden state size was 256, the number of layers was set to two with two LSTM cells on each layer. The initial learning rate was set to 5*10-3 and the dropout rate or keep probability was set to 0.75. We used a small subset of data for experimentation purpose containing 100,000 values in which the text length was in range 25-31. If the loss was not updated for 3 consecutive update steps then early stopping would be applied. Per epoch 3 update checks were carried out. The learning rate decay was set to 0.95 and minimum learning rate was set to 5*10-2.
  
## References

1. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio ,ICLR 2015.  
2. Github textsum,Xin Pan (xpan@google.com, github:panyx0718), Peter Liu (peterjliu@google.com, github:peterjliu)
