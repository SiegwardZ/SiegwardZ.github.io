---
layout: post
title: RL Theory Paper Journal
date: 2025-04-28 12:34 +0800
categories: [Study, Math, Paper]
tags: [math,study,optimization,rl]
description: Journal of rl theory papers, including theorems, lemmas and informative propositions.
comments: false
math: true
---
## Policy Gradient Methods for Reinforcement Learning with Function Approximation
[url](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
### Basic Notations and Definitions

$$
V^{\pi}(s) = \mathbb{E}\big[ \sum_{t=0}^{\infty} \gamma^t r_t | \pi, s_0=s \big]
$$

$$
Q^{\pi}(s,a) = \mathbb{E}\big[ \sum_{t=0}^{\infty} \gamma^t r_t | \pi, s_0=s,a_0=a \big]
$$

$$
d^\pi_{s_0}(s) = \sum_{t=0}^{\infty}\gamma^t \Pr(s_t=s|s_0,\pi)
$$

### Theorem 1 (Policy Gradient Theorem)
$$
\nabla_\theta V^{\pi_\theta}(s_0) = \sum_s d^{\pi_\theta}_{s_0}(s) \sum_a \pi_\theta(s,a) \big[\nabla_\theta \log \pi(s,a) Q^{\pi_\theta}(s,a) \big]
$$

### Theorem 2 (Policy Gradient with Function Approximation)
Let 
$$
f_w :  \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R} 
$$
be an approximation to 
$$
Q^\pi
$$.

If $$f_w$$ satisfies the following optimality condition (least-squares fit)

$$
\sum_s d^{\pi_\theta}_{s_0}(s) \sum_a \pi_\theta(s,a)\big[Q^{\pi_\theta}(s,a) - f_w(s,a)\big]\nabla_w f_w(s,a) = 0
$$  

and the compatibility condition 



$$
\nabla_w f_w(s,a) = \nabla_\theta \log \pi_\theta(s,a)
$$

then 

$$
\nabla_\theta V^{\pi_\theta}(s_0) = \sum_s d^{\pi_\theta}_{s_0}(s) \sum_a \pi_\theta(a|s) \big[ \nabla_\theta \log \pi(a|s) f_w(s,a) \big]
$$

It can be proved easily, according to the two conditions.

*Note:*

The optimality condition is obvious, but why this compatibility condition?

We want to use 
$$
f_w
$$
to estimate state-value function properly, i.e. it must satisfy

$$
\sum_s d^{\pi_\theta}_{s_0}(s) \sum_a \pi_\theta(s,a)\big[Q^{\pi_\theta}(s,a) \nabla_\theta \log \pi_\theta(s,a)\big]=  \sum_s d^{\pi_\theta}_{s_0}(s) \sum_a \pi_\theta(s,a)\big[ f_w(s,a)\nabla_\theta \log \pi_\theta(s,a)\big] 
$$

which is equivalent to

$$
\sum_s d^{\pi_\theta}_{s_0}(s) \sum_a \pi_\theta(s,a)\big[Q^{\pi_\theta}(s,a)- f_w(s,a)\big] \nabla_\theta \log \pi_\theta(s,a) = 0
$$

We already know 
$$
f_w 
$$
satisfies the optimality condition. The optimality condition can imply the equation above, iff the following condition is satisfied,

$$
\nabla_w f_w(s,a) = \nabla_\theta \log \pi_\theta(s,a)
$$

which is the compatibility condition.






### Corollary 2.1
If the policy is parameterized as

$$
\pi_\theta(s,a) = \frac{\exp(\theta^T \phi_{s,a})}{\sum_{a^\prime}\exp (\theta^T \phi_{s,a^\prime})}
$$

Then the compatibility condition requires that

$$
\nabla_w f_w(s,a) = \nabla_\theta \log \pi_\theta(s,a) = \phi_{s,a} - \nabla_\theta \log  \sum_{a^\prime}\exp (\theta^T \phi_{s,a^\prime}) = \phi_{s,a} - \sum_{a^\prime} \pi(s,a^\prime) \phi_{s.a^\prime} = \overline{\phi_{s,a}}
$$

So the natural parameterizaton of $$f_w$$ is 

$$
f_w(s,a) = w^T \big[ \phi_{s,a} - \sum_{a^\prime} \pi(s,a^\prime) \phi_{s.a^\prime} \big]
$$

*Note:* In fact

$$
f_w(s,a) = \int \overline{\phi_{s,a}} \; dw= w^T \overline{\phi_{s,a}}  + C(s,a)
$$

$$C(s,a)$$
is an arbitrary function without dependency on 
$$w$$
, and we can drop this term in parameterization.

Additionally in this sense, it is better to think
$$
f_w
$$
as an approximation of advantage function, as
$$
\sum_a \pi(s,a) f_w(s,a) = 0 
$$.


## A Natural Policy Gradient


## On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift
[arxiv](http://arxiv.org/abs/1908.00261)
### Definition
MDP
$$
M = (\mathcal{S},\mathcal{A},P,r,\gamma,\rho)
$$
, where $$\mathcal{S}$$ and $$\mathcal{A}$$ are both finite spaces.

Reward function: $$r : \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$$

Discount factor  $$\gamma \in [0,1)$$

Deterministic policy $$\pi:\mathcal{S} \rightarrow \mathcal{A}$$

Stocastic policy $$\pi: \mathcal{S} \rightarrow \Delta (\mathcal{A})$$

The value function is defined as  

$$
V^\pi(s) \mathrel{:=}  \mathbb{E}\bigg[ \sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)| \pi, s_0=s \bigg]
$$

The value function under the initial state distribution $$\rho$$ is defined as 

$$
V^\pi(\rho) \mathrel{:=}  \mathbb{E}_{s_0 \sim \rho} \big[ V^\pi(s_0)\big]
$$

The action-value function (Q-value) function is defined as 

$$
Q^\pi (s,a) \mathrel{:=}  \mathbb{E} \bigg[
    \sum_{t=0}^{\infty} \gamma^t r(s_t,a_t) | \pi, s_0=s, a_0=a 
\bigg]
$$

The advantage function is defined as 

$$
A^\pi(s,a) \mathrel{:=} Q^\pi (s,a) - V^\pi(s)
$$

The optimization problem the agent seeks to solve is

$$
\underset{\pi}{\max}\; V^\pi(\rho)
$$

Policy parameterization classes:

+ *Direct parameterization*. 

For 
$$\theta \in \Delta(\mathcal{A})^{|S|}$$

$$
\pi_\theta(a|s) = \theta_{s,a}
$$ 

+ *Softmax parameterization*. 

For 
$$\theta \in \mathbb{R}^{|S||A|}$$

$$
\pi_\theta(a|s) = \frac{\exp (\theta_{s,a})}{\sum_{a^\prime \in \mathcal{A}} \exp (\theta_{s,a^\prime})}
$$

+ *Restricted parameterization*. 

Parametric classes 
$$\{\pi_\theta | \theta \in \Theta \}$$
that may not contain all stocastic policies, like neural policy classes.

The discouted state visitation distribution of a policy $\pi$ is defined as

$$
d_{s_0}^\pi(s) \mathrel{:=}(1-\gamma)\sum_{t=0}^{\infty}\gamma^t \Pr^\pi(s_t=s|s_0)
$$

$$
d_\rho^\pi(s) \mathrel{:=} \mathbb{E}_{s_0\sim \rho}[d_{s_0}^\pi(s)]
$$

### Lemma 3.2 (The performance difference lemma)
For all policies 
$$\pi,\pi^\prime$$ 
and states
$$s_0$$

$$
V^\pi(s_0) - V^{\pi^\prime}(s_0) = \frac{1}{1-\gamma}\mathbb{E}_{s\sim d^\pi_{s_0}}\mathbb{E}_{a \sim \pi(\cdot|s)} \big[ A^{\pi^\prime}(s,a)\big]
$$