---
layout: post
title: Proximal Algorithms
categories: [Study, Math]
tags: [math,study,optimization]
description: Notes about proximal algorithms.
comments: false
math: true
date: 2025-03-08 22:25 +0800
---

## 1. Introduction
### Proximal Operator
Let $$f:\mathbb{R}^n \rightarrow \mathbb{R} \cup \{+ \infty \} $$ be a closed proper convex function.

The proximal operator $$\textbf{prox}: \mathbb{R}^n \rightarrow \mathbb{R}^n $$ of $$f$$ is defined by 

$$
\textbf{prox}_f (v) = \underset{x}{\text{argmin}}(f(x) + (1/2)||x-v||_2^2)
$$

$$
\textbf{prox}_{\lambda f} (v) = \underset{x}{\text{argmin}}(f(x) + (1/2\lambda)||x-v||_2^2)
$$

### Separable Sum
If $$f$$ is fully separable, meaning that $$f(x) = \sum_{i=0}^n f_i (x_i) $$, then

$$
(\textbf{prox}_f (v))_i = \textbf{prox}_{f_i}(v_i) 
$$

### Fixed Points
The point $$x^*$$ minimizes $$f$$ if and only if $$x^* = \textbf{prox}_f(x^*)$$

### Moreau Decomposition
The following relation always holds:

$$
v = \textbf{prox}_f(v) + \textbf{prox}_{f^*}(v)
$$

where $$f^*(y)$$ is the convex conjugate of $$f$$.

proof: 

$$
\begin{align*}
x = \textbf{prox}_f(v) &\iff x = \underset{z}{\text{argmin}}[f(z) + (1/2)||z-v||_2^2] \\
& \iff 0 \in \partial f(x) + (x-v) \\
& \iff v-x \in \partial f(x) \\
& \iff x \in \partial f^*(v-x) \;\; \text{(property of gradient in Fenchel conjugate)}\\ 
& \iff v - (v-x) \in \partial f^*(v-x) \\
& \iff v- x = \textbf{prox}_{f^*}(v) \\
& \iff v = \textbf{prox}_f(v) + \textbf{prox}_{f^*}(v)
\end{align*}
$$


**Extended Moreau Decomposition**
For any $$\lambda \gt 0$$, 

$$
v = \textbf{prox}_{\lambda f}(v) + \lambda \textbf{prox}_{\lambda^{-1} f^*}(v/\lambda)
$$

proof: to be checked


**ref:** [quanquan gu notes](https://piazza.com/class_profile/get_resource/is58gs5cfya7ft/iw7c556bwdf4vt)

## 2. Moreau-Yosida regularization (Moreau Envelope)
### Definition
The infimal convolution of closed proper convex functions $$f$$ and $$g$$ on $$R^n$$, denoted $$f \Box g$$, is defined as 

$$
(f \Box g)(v) = \underset{x}{inf}(f(x)+g(v-x))
$$

with $$\textbf{dom}(f \Box g) = \textbf{dom} f + \textbf{dom} g$$.

The main example is *Moreau envelope* or *Moreau-Yosida regularization* $$ M_{\lambda f} = (\lambda f) \Box (1/2) \|\cdot \|_2^2 $$, *i.e.*,


$$
M_{\lambda f} (v) = \underset{x}{inf}(f(x) + (1/2\lambda)||x-v||_2^2)
$$


### Properties (to be proved)
**Property 2.1**
Moreau-Yosida regularization $$M_f$$ is essentially a smoothed or regularized form of $$f$$:
1. It has domain $$R^n$$
2. It is continuously differentialble
3. The sets of minimizers of $$f$$ and $$M_f$$ are the same. [hint for proof](https://math.stackexchange.com/questions/75532/moreau-yosida-regularization-problem)

**Property 2.2**
Relationship between proximal operator and Moreau-Yosida regularization. $$\textbf{prox}_f(x)$$ returns the unique point that actually achieves the infimum that defines $$M_f$$,

$$
M_f(x) = f(\textbf{prox}_f(x)) + (1/2)||x-\textbf{prox}_f(x)||_2^2
$$

**Property 2.3**
Gradient of Moreau-Yosida regularization is given by

$$
\nabla M_{\lambda f}(x) = (1/\lambda)(x-\textbf{prox}_{\lambda f}(x))
$$

## refs
+ https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf
+ https://web.stanford.edu/class/ee364b/lectures.html
+ https://stanford.edu/~boyd/papers/pdf/prox_slides.pdf
+ https://www.stat.cmu.edu/~ryantibs/convexopt/
+ https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

## other related topics
+ monotone operator theory
+ fixed point theory
+ subgradient and subdifferential
+ dual norm explanation
    + https://math.stackexchange.com/questions/903484/dual-norm-intuition