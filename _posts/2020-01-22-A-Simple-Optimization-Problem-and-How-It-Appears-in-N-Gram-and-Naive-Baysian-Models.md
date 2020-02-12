---
layout: post
title: A Simple Optimization Problem and How It Appears in N-Gram and Naive Baysian Models
categories: machine learning
published: true
---


The following optimization problem appears in many applications and is easy to solve.

$$\begin{align}
 \max_{x_1,...,x_n} & \sum_{i=1}^n c_i \log x_i \tag{1} \\
 s.t. & \sum_{i=1}^n x_i = b
\end{align}$$
, where $$c_1,...,c_n > 0$$. The problem is equivalent to 
 
 $$\begin{align}
 \min_{x_1,...,x_n} & - \sum_{i=1}^n c_i \log x_i \tag{2} \\
 s.t. & \sum_{i=1}^n x_i = b
\end{align}$$
. This is a convex optimization problem, and can be solved by finding its KKT conditions. The Lagrange dual function of the above problem is

$$L(x_1,...,x_n) = -\sum_{i=1}^n c_i \log x_i + \alpha (\sum_{i=1}^n x_i - b)$$

. Taking the partial derivatives we get

$$ \frac{\partial L}{\partial x_i} = -\frac{c_i}{x_i} + \alpha$$

. The KKT conditions are 

$$\begin{align}
\frac{\partial L}{\partial x_i} = -\frac{c_i}{x_i} + \alpha &= 0, \ i=1,...,n \\
\sum_{i=1}^n x_i &= b
\end{align}$$

. Solving the above equations yields

$$x_i = \frac{c_i}{c} \cdot b \tag{3} $$

for $$i=1,...,n$$, and we let $$c = \sum_{i=1}^n c_i$$.

The solution given by Equation 3 is easily interpreted: $$x_i$$ is propotional the ratio $$\frac{c_i}{c}$$. We will see how this problem appears in both N-Gram Language Model and Naive Baysian Model.
Their solutions are often interpreted as "relative frequency".

## N-Gram Language Model
A language model is a statistical model that gives a probability distribution that a document occurs. A document is often treated as a sequence of words. So 
the probability that a document occurs is the joint probability that its words co-occur. In order to train such a model, we are typically given a training sample
of $$K$$ documents. We assume these documents are drawn i.i.d from the target probability distribution, and the model is trained by the principle of maximum likelihood.

To be specific, let's assume a vocabulary $$V$$. For a document of $$m$$ words, let $$X_i, \ i=1,...,m$$ be a random variable that takes one of the $$V$$ words as value. Then a language model
aims to give a joint (and marginal) probability of 

$$P(X_1,...,X_m)$$

. Note that the sample set of the probability space is the coutable Cartesian product $$ V \times V \times \cdots $$

The N-Gram language model assumes a specific form of the joint probability distribution: 

$$P(X_1,...,X_m) = P(X_m|X_1,...,X_{m-1})P(X_{m-1}|X_1,...,X_{m-2})\cdots P(X_1)$$

. The probability of the word $$X_i$$ depends only on its previous $$n-1$$ words $$X_{i-1}, ..., X_{i-n+1}$$. That is, the conditional probability

$$P(X_i|X_{i-1},...,X_1) = P(X_i|X_{i-1},...,X_{i-n+1})$$

. As a result, we split the sequence of $$m$$ words into $$m-n+1$$ groups of $$n-$$grams, and the joint probability is decomposed to $$m-n+1$$ conditional probabilities

$$P(X_1,...,X_m) = P(X_m|X_{m-1},...,X_{m-n+1}) \cdot P(X_{m-1}|X_{m-2},...,X_{m-n}) \cdots P(X_{n}|X_{n-1},...,X_1) \cdot P(X_1,...,X_{n-1})$$ 

In order to train a N-Gram model, we are typically given $$K$$ documents, each of which is assumed to drawn i.i.d. from the target distribution. So the joint distribution of these $$K$$ training
documents is the product of $$K$$ joint probabilities. Since each joint probability is a product of $$n-$$grams conditional probabilities, the joint probability of the $$K$$ documents can be written as

$$ P(K \ training \ samples) = \big[\prod_{i=1}^{M} P_i(X_n|X_{n-1},...,X_1) \big]  \cdot \big[ \prod_{j=1}^K P_j(X_1,...,X_{n-1})\big]

$$

, where $$M$$ is the total number of $$n$$-grams. 

This model has a total number of $$|V|^{n-1}+|V|^n$$ parameters. The first set of parameters are the joint probabilities of $$n-1$$-grams, while the second conditional probabilities of $$n$$-grams.
The model is trained by the principle of maximum likelihood. We aim to find the set of optimal parameters that maximizes $$P(K \ training \ sample)$$. It is equivalent to maximize its logarithm, and thus
the associated optimization problem becomes 

$$\begin{align}
 \max & \ \ \log \Big( \big[\prod_{i=1}^{M} P_i(X_n|X_{n-1},...,X_1) \big]  \cdot \big[ \prod_{j=1}^K P_j(X_1,...,X_{n-1})\big] \Big) \\
 s.t. & \sum_{X_1,...,X_n \in V^{n-1}} P(X_1,...,X_{n-1}) = 1 \\
      & \sum_{X_n \in V} P(X_n|X_{n-1},...,X_1) = 1, \ for \ any \ combination \ of \ X_1,...,X_{n-1}
\end{align}$$

. The objective function of the above optimization problem can be decomposed into two parts:

$$\sum_{i=1}^M \log P_i(X_n|X_{n-1},...,X_1) + \sum_{j=1}^K \log P_j(X_1,...,X_{n-1}) \tag{3} $$

. Each part can be optimized independently. For the first part, we can group the summands first by their $$n-1$$ conditions $$X_1,...,X_{n-1}$$, and then by the value of $$X_n$$. The summation becomes

$$\sum_{X_1,,...,X_{n-1} \in V^{n-1}} \sum_{w \in V} c_w \log P(X_n = w|X_{n-1},...,X_1) $$

, where $$c_w$$ is the number times where $$X_n = w$$ for a specific combination of $$X_1,...,X_{n-1}$$. 
Again the $$|V|^{n-1}$$ parts above can be optimized independently. Writing down the optimization problem for each part, we get

 $$\begin{align}
 \max & \ \ \sum_{w \in V} c_w \log P(X_n = w|X_{n-1},...,X_1) \tag{4} \\
 s.t. & \sum_{w \in V} P(X_n=w|X_{n-1},...,X_1) = 1, 
\end{align}$$

. Problem (4) is an instance of Problem (1) where $$b=1$$. So the solutions are 
$$P(X_n=w | X_{n-1},...,X_1) = \frac{c_w}{c} \tag{5} $$

, where $$c = \sum_{w \in V} c_w$$ is the count of occurences where the combination $$X_1,....,X_{n-1}$$ occurs. So the solution in Equation (5) states that the conditional probability
 $$P(X_n=w|X_{n-1},...,X_1)$$ is its relative frequency: out of the $$c$$ occurences of $$X_1,...,X_{n-1}$$, the word $$w$$ occurs $$c_w$$ times.
 
The optimization problem of the second part of Equation (3) is

 $$\begin{align}
 \max & \ \ \sum k_{X_1,...X_{n-1}} \log P(X_1,...,X_{n-1}) \tag{6} \\
 s.t. & \sum_{X_1,...,X_{n-1} \in V^{n-1}} P(X_1,...,X_{n-1}) = 1, 
\end{align}$$ 

, where $$k_{X_1,...,X_{n-1}}$$ is the number of occurences of the combination $$X_1,...,X_{n-1}$$ in the training sample. Again this is an instance of Problem (2), and the solutions are

$$P(X_1,...,X_{n-1}) = \frac{k_{X_1,...,X_{n-1}}}{K}$$

.

N-Gram is a generative model by which we can generate a document. In practice, we often utilize the trained conditional probability $$P(X_n|X_{n-1},...,X_1)$$ to predict the $$n-$$word 
given its preceding $$n-1$$ words.

## Naive Baysian Model

The Naive Baysian model is often applied in many classification problems. In these problems, we would like to obtain the conditional probability $$P(X=x|Y=y)$$, based on which the classifier function 
$$y=f(x)$$ is obtained via 

$$\hat{y} = \max_y P(Y=y|X=x)$$

. By Baysian Theorem, the conditional probabiilty

$$P(Y=y|X=x) = \frac{P(X=x, Y=y))}{P(X=x)}$$

. Thus, given input $$X=x$$, if we know the joint probabiilty $$P(X=x, Y=y)$$ for each possible $$y$$, then we can make the classification that gives the greatest conditional probabiilty.

The joint distribution of the feature space $$X$$ and target domain $$Y$$ can be decomposed to $$P(X, Y) = P(Y) \cdot P(X|Y)$$. The Naive Baysian model assumes that the probability distributions 
of features are independent given a specific $$y$$. That is, the conditional probability

$$P[X=(x_1,...,x_n)|Y=y] = P(x_1|y)\cdot P(x_2|y) \cdots P(x_n | y)$$

. Therefore, given a training sample of $$K$$ instances, assuming these instances are drawn i.i.d., the joint probability is

$$\prod_{i=1}^K P(X^{(i)}, Y^{(i)}) = \prod_{i=1}^K P[Y^{(i)}] \cdot P[x_1^{(i)}|Y^{(i)}] \cdots P[x_n^{(i)}|Y^{(i)}] $$

. To maximize the above joint probabiilty is equivalent to maximize its logarithm. Therefore, the associated optimization problem is

 $$\begin{align}
 \max & \ \ \sum_{i=1}^K \log P[Y^{(i)}]  + \sum_{i=1}^K (\log P[x_1^{(i)}|Y^{(i)}] + \log P[x_2^{(i)}|Y^{(i)}]  + ... + \log P[x_n^{(i)}|Y^{(i)}])\tag{7} \\
 s.t. & \sum_{y} P[Y=y] = 1 \\
 & \sum_{x \in D_j} P(x_j = x | Y=y) = 1, \ j=1,...,n
\end{align}$$ 
, where $$D_j$$ is the domain of the $$j-$$th feature.

The two parts in (5) can be optimized independently. The objective function of the first part can be written as 

$$\sum_{i=1}^K \log P[Y^{(i)}] = \sum_{j=1}^M c_j \log P[Y=y_j]$$

, where $$c_j$$ is the number of occurences of $$Y=y_j$$ in the traning sample. Thus the associated optimization problem in the first part is:

 $$\begin{align}
 \max & \ \ \sum_{j=1}^M c_j \log P[Y=y_j]  \tag{8} \\
 s.t. & \sum_{j=1}^M  P[Y=y_j] = 1 	
\end{align}$$ 

Again this is an instance of Problem (1), and solutions are $$P[Y=y_j] = \frac{c_j}{c}$$, where $$c = \sum_{j=1}^M c_j$$.

For the second part in (5), the $$n \cdot K$$ summands can be divided to $$n$$ groups. Each feature forms a group. And these $$n$$ groups can be optimized independently.
For each group, the summantion can be further grouped by $$Y=y_j$$. Eventually, the second part is divided into $$n \cdot M $$ independent optimization problems, each of which is

 $$\begin{align}
 \max & \ \ \sum_{h=1}^{N_d} c_h \log P[x_d = v_h |Y=y_j] \\
 s.t. & \sum_{h=1}^{N_d}  P[x_d = v_h | Y=y_j] = 1 	
\end{align}$$ 

, where $$x_d = v_h$$ means the $$d-$$feature takes value $$v_h$$, and $$c_h$$ is the number of occurences of $$x_d = v_h$$ in the training sample. Needless to say, these optimization problems
are instances of Problem (1). So the solutions are 

$$P(x_d = v_h | Y=y_j) = \frac{c_h}{c_{dj}}$$

, where $$c_{dj}$$ is the number of occurences of $$Y=y_j$$ for the $$d-$$the feature.