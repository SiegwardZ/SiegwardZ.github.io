---
layout: post
title: Useful Optimization Results
categories: [Study, Math, Optimization]
tags: [math,study,optimization]
description: Some optimization results may be useful.
comments: false
math: true
date: 2025-04-29 15:40 +0800
---

## Theroem 1
Theorem 14.8 and Lemma 14.9 from Shalev-Shwartz and Ben-David *Understanding Machine Learning: From Theory to Algorithms.*

Consider the stocastic projected gradient descent algorithm where at each step $$t$$, 
we update $$x_t$$ by sampling a random
$$v_t$$
such that

$$
x_{t+1} = P_C(x_t - \eta v_t)
$$

where 
$$
\mathbb{E}[v_t|x_t] = \nabla f(x)
$$

Assume 
$$C = \{x:\|x\| \le B\}$$
, for some 
$$B\gt 0$$
Let $$f$$ be a convex function and let $$x^* \in \text{argmin}_{x:\|x\| \le B}f(w) $$.
Assume also that for all $$t$$, $$\|v_t\| \le \rho$$, and that stocastic projected gradient descent is run for $$N$$ iterations with $$\eta=\sqrt{\frac{B^2}{\rho^2N}}$$.
Then,

$$
\mathbb{E} \bigg[f\bigg(\frac{1}{N} \sum_{t=1}^{N} x_t \bigg) \bigg] - f(x^*) \le \frac{B\rho}{\sqrt{N}}
$$

