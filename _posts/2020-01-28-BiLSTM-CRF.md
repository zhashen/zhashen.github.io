---
layout: post
title: Some Notes on Conditioanl Random Fields
categories: machine learning
published: true
---

Sequence tagging is a common task in Natural Language Processing. In this task, we are typically given a sequence of words $$x_1,...,x_n$$. We would like to tag each word, and thus generate a sequence
of tags $$y_1,...,y_n$$. The simple linear chain CRF models the joint distribution of the two sequences as

$$\begin{align}
p(x_1,...,x_n, y_1,...,y_n) &= [p(y_1)p(x_1|y_1)] \cdot [p(y_2|y_1)p(x_2|y_2)] \cdots [p(y_n|y_{n-1})p(x_n|y_n)] \\
&= \prod_{i=1}^n [p(y_i|y_{i-1})p(x_i|y_i)] 
\end{align}$$

, where we let $$p(y_1 \mid y_0) = p(y_1)$$. The above model assumes "local factoring": the random variable $$y_i$$ only depends on $$y_{i-1}$$, and $$x_i$$ only depends on $$y_i$$.

The components in the above equation can be written as

$$p(y_i|y_{i-1})p(x_i|y_i) = e^{\log [ p(y_i|y_{i-1})p(x_i|y_i) ]} = e^{\log p(y_i|y_{i-1}) + \log p(x_i|y_i)} \tag{1}$$

Therefore, if we allow more generic forms for the exponents, then 

$$p(y_i|y_{i-1})p(x_i|y_i) \propto e^{a_i + b_i} $$

. Thus

$$p(x_1,...,x_n, y_1,...,y_n) \propto \prod_{i=1}^n e^{a_i + b_i} = e^{\sum_{i=1}^n (a_i+b_i)}$$

. We can think of $$\sum_{i=1}^n (a_i + b_i)$$ as a scoring function $$S(x_1,...,x_n, y_1,...,y_n)$$. Then we say that the joint probability $$p(x_1,...,x_n, y_1,...,y_n)$$ is proportional
to a score on $$x_1,...,x_n, y_1,...,y_n$$. This score is a sum of $$n$$ components. Each component consists of a score that determines $$x_i$$ given $$y_i$$, and a score that determines $$y_i$$ 
given $$y_{i-1}$$. We call the former "emit score", and the latter "transition score".

The complete form of the joint probability distribution is

$$p(x_1,...,x_n, y_1,...,y_n) = \frac{1}{Z} \cdot e^{S(x_1,...,x_n, y_1,...,y_n)} \tag{2} $$

, where $$Z = \sum_{x_1,...,x_n, y_1,...,y_n} S(x_1,...,x_n, y_1,...,y_n)$$ is a normalization factor. So Equation (1) is merely a special case of Equation (2), where we let $$a_i = \log p(x_i \mid y_i)$$, 
and $$b_i = \log p(y_i \mid y_{i-1})$$.

## BiLSTM-CRF

The BiLSTM-CRF model is a Sequence-to-Sequence model that achieves very good performance in practice. In a typical implementation of BiLSTM-CRF, the input sequence of $$n$$ words first pass through 
an embedding layer, and then a BiLSTM layer. The output length of the BiLSTM layer is still $$n$$, but each element has the same dimension as the tag size $$m$$. In other words, we have a $$m \times n$$ 
matirx 

$$ A = 
\begin{pmatrix}
a_{11} & \cdots & a_{1n} \\
a_{21} & \cdots & a_{2n} \\
\vdots & \cdots & \vdots \\
a_{m1} & \cdots & a_{mn} \\
\end{pmatrix}$$

, where $$a_{ij}$$ is the emit score from tag $$j$$ to the input $$x_i$$. Each column vector of $$A$$ is the vector that stores the emit scores from each tag to the input word at that step.

After that, the CRF takes the emit vectors as input, and outputs the final sequence $$y_1,...,y_n$$. The CRF layer is parameterized by a transition matrix $$B = [b_{ij}]$$, where each entry $$b_{ij}$$ 
is the transition score from tag $$i$$ to tag $$j$$.

To train the CRF layer, we are given a sample of $$K$$ instances. We would like to minimize their training errors. The error function of each instance is its negative log-likelihood,


$$\begin{align}
L(x, y) &= -\log p(y|x) \\
&= -\log \frac{p(x, y)}{p(x)}  \\
&= -\log \frac{\frac{1}{Z}e^{S(x, y)}}{\frac{1}{Z}\sum_{y'} e^{S(x, y')}} \\
&= \log [\sum_{y'} e^{S(x, y')}] - S(x, y)
\end{align}$$

Given $$A, B, x, y$$, the second term $$S(x, y)$$ is straightforward to obtain. The difficulty lies in evaluating the first term $$\log [\sum_{y'}e^{S(x, y')}]$$. Here $$y'$$ means a possible 
path $$y_1,...,y_n$$. There are totally $$m^n$$ paths, and it is too costly to evaluate all of them. 

The trick is to use Viterbi algorithm. Let $$(*, y_n=j)$$ denote a path of which the $$n$$-th step is $$j$$. Similarly, $$(*, y_{n-1})$$ means a path of which the $$(n-1)$$-th 
step is $$j$$. Now the first term can be written as


$$\begin{align}
\log [\sum_{y'} e^{S(x, y')}] &= \log \big( e^{S_{1,1,...,1}} + ... + e^{S_{m,m,....m}} \big), \ m^n \ terms \ , \ s_{...} \ is \ the \ score \ of \ a \ path \\
&= \log \big(\sum_{*}e^{S(*, y_n=1)} + ... + \sum_{*}e^{S(*, y_n=m)} \big), \ the \ sum \ is \ divided \ into \ m \ groups \\
&= \log \big( e^{H(n-1, 1)} + ... + e^{H(n-1, m)}\big)
\end{align}$$

, where $$e^{H(n-1, j)} = \sum_{*}e^{S(*, y_n=j)}$$. The idea is to compute $$H(n-1, j)$$ in a recursive way. That is, $$H(n-1, j) = f(H(n-2, 1),...,H(n-2, m))$$. 
To find out such recursive function, observe

$$\sum_{*} e^{S(*, y_n=j)} = e^{\log (\sum_{*} e^{S(*, y_n=j)} )}$$

. Therefore

$$\begin{align}
H(n-1, j) &= \log (\sum_{*} e^{S(*, y_n=j)}) \\
&= \log \big[ (e^{S(*, y_{n-1}=1, y_n=j)} + ... ) + ... + (e^{S(*, y_{n-1}=m, y_n=j)}+...)\big] , \ the \ sum \ is \ divided \ into \ m \ groups \\
&= \log \big[ e^{a_{jn}+b_{1j}} e^{H(n-2, 1)}+ ... +  e^{a_{jn}+b_{mj}} e^{H(n-2, m)} \big]
\end{align}$$

. Note that in the above equation $$e^{H(n-2, j)} = \sum_{*} e^{S(*, y_{n-1}=j)}$$.

So we have found the recursive function, and the initial conditions are $$H(1, j) = e^{a_{j1} + b_{j}}$$. Understanding the recursive way of computing $$H(n-1, j)$$ is key to understanding
the implementation of BiLSTM-CRF model, e.g. [in this tutorial](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html). 