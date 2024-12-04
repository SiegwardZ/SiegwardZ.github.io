---
layout: post
title: Convex Optimization - Chapter 0
categories: [Study, Math]
tags: [math,study,optimization]
description: Introduction and Mathematical Preliminaries for Convex Optimization
comments: false
math: true
date: 2024-11-27 22:16 +0800
---
## Norms

---

### Basic Definitions
#### **standard inner product**

For $x,y \in \mathbb{R}^n$

$$
<x,y> = x^T y = \sum_{i=1}^n x_i y_i
$$

For $X,Y \in \mathbb{R}^{m\times n}$

$$
<X,Y> = tr(X^TY) = \sum_{i=1}^m \sum_{j=1}^n X_{ij}Y_{ij}
$$

#### **Norm**

A function $f:\mathbb{R}^n \rightarrow \mathbb{R} $ with $dom f = \mathbb{R}^n$ is called a norm if
+ $f$ is nonnegative: $f(x) \ge 0$ for all $x\in \mathbb{R}^n$
+ $f$ is definite: $f(x)=0$ only if $x=0$
+ $f$ is homogeneous: $f \(t x\) = \|t\| f \(x\) $, for all $ x\in \mathbb{R}^n, t\in \mathbb{R}$
+ $f$ satisfies the triangle inequality: $f(x+y) \le f(x) + f(y)$ for all $x,y\in \mathbb{R}^n $


#### **distance**

For two vectors x,y

$$
dist(x,y) = \|x-y\|
$$

#### **unit ball**

The set
$$
B = \{x\in  \mathbb{R}^n | \; \|x\| \le 1 \}
$$
is called the *unit ball* of the norm $$\| · \|$$ 
- $B$ is symmetric about the origin, i.e. $x\in B \iff -x \in B$
  - homogeneous
- $B$ is convex
  - triangle inequality
- $B$ is closed, bounded, and has nonempty interior


#### **Euclidean norm / $\ell_2$-norm**

$$
\|x\|_2 = (x^T x)^{\frac{1}{2}} = (x_1^2 + x_2^2 + \cdots + x_n^2)^{\frac{1}{2}}
$$

#### **Cauchy-Schwartz inequality**

$$
|x^Ty| \le \|x\|_2\|y\|_2
$$

#### **(unsigned)angle**

between nonzero vectors x,y

$$
\angle (x,y) = cos^{-1}(\frac{x^Ty}{\|x\|_2\|y\|_2}) \in [0,\pi]
$$

We say x and y are orthogonal if $x^Ty=0$.

#### **Frobenius norm**

For a matrix $X \in \mathbb{R}^{m\times n}$

$$
\|X\|_F = (tr(X^TX))^{\frac{1}{2}} = (\sum_{i=1}^m \sum_{j=1}^n X^2_{ij})^{\frac{1}{2}}
$$

#### **$\ell_1$-norm**

$$
\|x\| = |x_1| + |x_2| + \cdots + |x_n|
$$

#### **Chebyshev norm / $\ell_\infty$-norm**

$$
\|x\|_\infty = max\{|x_1|,\ldots,|x_n|\}
$$

#### **$\ell_p$-norm ($p\ge 1$)**

$$
\|x\|_p = (|x_1|^p + \cdots + |x_n|^p )^{\frac{1}{p}}
$$

#### **quadratic norms**

For $P \in \mathbb{S}^n_{++} $,  the $P$-quadratic norm is defined as 

$$
\|x\|_P = (x^TPx)^{1/2} = \|P^{1/2}x\|_2
$$

The unit ball of a quadratic norm is an ellipsoid, and vice versa.

---

### Equivalence of norms

$$\|·\|_a, \|·\|_b$$ are norms on $\mathbb{R}^n$

$$
\begin{align*}
\|·\|_a \text{ and } \|·\|_b\& \text{ are equivalent} \\ 
&\iff \\
\text{there exists postivie constants }& \alpha, \beta \text{ such that, for all } x\in \mathbb{R}^n \\
\alpha \|x\|_a \le &\|x\|_b \le \beta\|x\|_a
\end{align*}
$$

---

### Operator norms

$$\|·\|_a, \|·\|_b$$ are norms on $\mathbb{R}^m, \mathbb{R}^n$, respectively. The *operator norm* of $X\in \mathbb{R}^{m \times n}$ is defined as 

$$
\|X\|_{a,b} = sup\{\|Xu\|_a \;|\; \|u\|_b \le 1\}
$$

$u$ is any vector in the unit ball of norm-b. Calculate norm-a on the vectors obtained by multiplying $X$ with $u$. The supremum is $\|X\|_{a,b}$

#### **spectral norm / $\ell_2$-norm of $X$**

$$
\|X\|_2 = \|X\|_{2,2} = \sigma_{max}(X) = (\lambda_{max}(X^T X))^{1/2}
$$

#### **max-row-sum norm / $\ell_\infty$-norm of $X$**

$$
\|X\|_\infty = sup\{\|Xu\|_\infty \;|\; \|u\|_\infty \le 1\} = \underset{i=1,2,\ldots,m}{max} \sum_{j=1}^n |X_{ij}|
$$

#### **max-column-sum norm / / $\ell_1$-norm of $X$**

$$
\begin{equation*}
\|X\|_1 = sup\{\|Xu\|_1 \;|\; \|u\|_1 \le 1\} = \underset{j=1,2,\ldots,n}{max} \sum_{i=1}^m |X_{ij}|
\end{equation*}
$$

note:

$$
\begin{align*}
\|u\|_1 \le 1 & \iff |u_1| + \cdots  + |u_n| \le 1 \\
\|Xu\|_1 &= (|X_{11}u_1| + |X_{12}u_2| + \cdots + |X_{1n}u_n|  ) \\
  & + \ldots \\
  & + (|X_{m1}u_1| + |X_{m2}u_2| + \cdots + |X_{mn}u_n|  ) \\
  & = (|X_{11}|+|X_{21}| + \cdots + |X_{m1}|)|u_1| \\
  & + \ldots \\
  & + (|X_{1n}| + |X_{2n}| + \cdots + |X_{mn}|)|u_n| \\
  & \le \underset{j=1,2,\ldots,n}{max} \sum_{i=1}^m |X_{ij}| (|u_1| + \cdots + |u_n|) \\
  & \le \underset{j=1,2,\ldots,n}{max} \sum_{i=1}^m |X_{ij}|
\end{align*}
$$

---

### Dual norm

Let $$\|·\|$$ be a norm on $\mathbb{R}^n$. The associated dual norm $$\|·\|_*$$ is defined as 

$$
\|z\|_* = sup\{z^T x \;|\; \|x\| \le 1\}
$$

It can be interpreted as the operator norm of matrix $z^T \in \mathbb{R}^{1\times n}$, with the norm $$\|·\|$$ on $\mathbb{R}^n$ and the absolute value on $\mathbb{R}^1$.

#### **Prove that $$\|·\|_*$$ is a norm**
+ nonnegative and definite

It is obvious that $$\|\mathbf{0}\| = 0$$.

For any $z \in \mathbb{R}^n \ne \mathbf{0}$, take $$x = \frac{z}{\|z\|}$$, then $$\|z\|_* \ge z^Tx = \frac{\|z\|_2^2}{\|z\|} \gt 0 $$. Note that $$\|\frac{z}{\|z\|}\| = \frac{1}{\|z\|} \|z\| = 1$$


+ homogeneous

Note that 

$$\|z\|_{*}$$ 
$$= sup\{ | z^T x | \;|\; \|x\| \le 1\}$$, 

so for any $t \in \mathbb{R}$, we have $$\|tz\|_*$$
$$ = sup\{|t z^T | \;|\; \|x\| \le 1\} $$
$$ = |t| sup\{|z^T x| \;|\; \|x\| \le 1\} $$
$$ = |t|\|z\|_* $$

+ triangle inequality

For any $z_1,z_2 \in \mathbb{R}^n$, we have
$$\|z_1+z_2\|_* $$
$$ = sup\{|(z_1^T + z_2^T) x| \;|\; \|x\| \le 1\}$$
$$ \le sup\{|z_1^T x| + |z_2^T x| \;|\; \|x\| \le 1\}$$
$$ = sup\{|z_1^T x| \;|\; \|x\| \le 1\} + sup\{|z_2^T x| \;|\; \|x\| \le 1\}$$
$$ = \|z_1\|_* + \|z_2\|_*$$


#### **an inequality**

For any $y,z \in \mathbb{R}^n$,

$$
z^Ty \le \|y\|\|z\|_*
$$

Note that 
$$\|y\|\|z\|_* = \|y\| sup\{z^Tx \;|\; \|x\| \le 1\}$$
$$ \ge \|y\|z^T\frac{y}{\|y\|} = z^Ty$$

#### **dual of dual norm**

In finite-dimensional vector spaces, the dual of the dual norm is the original norm.

My thoughts:

We have 

$$
\|z\|_{**} = sup\{z^Tx \;|\; \|x\|_* \le 1\}
$$

First, we can easily prove that $$\|z\|_{**} \le \|z\|$$: for any $x$, we have $$\|z\| \ge \frac{x^Tz}{\|x\|_*} \ge z^Tx$$

Next we only need to prove that this upper bound can be obtained. That is, there exists $x$ with $$\|x\|_* \le 1$$, such that $$\|z\|=z^Tx$$.

Then, it may need some knowledge that I haven’t learned yet. See [ref](https://math.stackexchange.com/questions/540020/dual-norm-of-the-dual-norm-is-the-primal-norm)

#### **Special examples**
+ The dual of the Euclidean norm is the Euclidean norm

From Cauchy-Schwarz's inequality, $$z^Tx \le |z^Tx| $$
$$\le \|z\|_2\|x\|_2 \le \|z\|_2 $$,
 and the value of 
$$x$$ 
that maximizes $$z^Tx$$ 
is 
$$\frac{z}{\|z\|_2}$$
 

## 2

