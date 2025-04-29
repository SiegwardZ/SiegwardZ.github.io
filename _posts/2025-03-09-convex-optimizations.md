---
layout: post
title: Convex Optimizations
categories: [Study, Math, ConvexOpt]
tags: [math,study,optimization,convex optimization]
description: Notes about convex optimizations
comments: false
math: true
date: 2025-03-09 12:43 +0800
---

## 1. Convex Conjugate and Fenchel Conjugate
###  Definitiion
For function $$f$$, the Fenchel conjugate is defined to be 

$$
f^* (y) = \underset{x \in dom(f)}{\text{sup}}(y^Tx - f(x))
$$

An important property: whether $$f$$ is convex or not, $$f^*$$ is convex. This is because $$f^*$$ is the pointwise maximun/supermum of affine functions.

### Fenchel's inequality (Fenchel-Young inequality)
By definition, we can get Fenchel's inequality,

$$
f^*(y) \ge x^Ty-f(x)
$$

**Propositions 1.1**

We notice that for any $$y$$ by Fenchel's inequality,

$$
f(x) \ge x^Ty -f^*(y)
$$

so that,

$$
f(x) \ge \underset{y}{\text{sup}} [x^Ty - f^*(y)] = f^{**}(x)
$$

**Propositions 1.2** 

$$
f^*(y) + f(x) = x^Ty \iff y \in \partial f(x)
$$

Indeed, we have

$$
\begin{align*}
f^*(y) + f(x) = x^Ty & \iff f(x) + z^Ty - f(z) \le x^Ty, \forall z \\
& \iff f(x) + y^T(x-z) \le f(z), \forall z \\
& \iff y \in \partial f(x) 
\end{align*}
$$

**Proposition 1.3**

If $$f$$ is convex, then
$$
f = f^{**}
$$.

By Fenchel's inequality, we already have
$$
f \ge f^{**}
$$

By definition, we have

$$
f^{**}(x) = \underset{y}{sup}[y^T x - f^*(y)] \ge y^Tx - f^*(y), \forall y
$$

As $$f$$ is convex, there exists $$y\in \partial f(x)$$, such that 

$$
y^Tx - f^*(y) = f(x)
$$

Then $$f^{**}(x) \ge y^Tx - f^*(y) = f(x)$$. Then $$f = f^{**}$$.

### Inverse of Gradient in Fenchel conjugate
Suppose that $$f$$ is closed and convex, then $$\partial f$$ and $$\partial f^*$$ are inverses in the following sense:

$$
y \in \partial f(x) \iff x \in \partial f^*(y)
$$

Proof to be checked: $$f(x)$$ and $$f^*(y)$$ are symmetrical, similar to the proof in proposition 1.2


### ref 
+ [cmu convexopt lec12](https://www.stat.cmu.edu/~siva/teaching/725/lec12.pdf)
+ [cmu convexopt lec13 ](https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/13-dual-corres-scribed.pdf)
+ [stanford ee364b subgradient notes](https://web.stanford.edu/class/ee364b/lectures/subgradients_notes.pdf)
+ [cmu convexopt lec4](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/lec4.pdf)
+ [tmu notes](https://cvg.cit.tum.de/_media/teaching/ss2015/multiscale_methods/board1.pdf)

