# Chapter 2: Linear Algebra

## 2.1 Scalars, Vectors, Matrices and Tensors
+ **Scalars**: A scalar is just a single number.
    + Notation: $s \in \mathbb{R}$.
+ **Vectors**: A vector is an array of numbers.
    + Notations
        + vector: $\boldsymbol{x}$.
        + element: $x_1$.
        + set: $S = \{1, 3, 6\}$, $\boldsymbol{x}_{S} = \{x_1, x_3, x_6\}$.
        + complement: $\boldsymbol{x}_{-S}$ is the vector containing all of the elements of $\boldsymbol{x}$ except for $x_1$, $x_3$ and $x_6$.
+ **Matrices**: A matrix is a 2-D array of numbers.
    + Notations
        + matrix: $\boldsymbol{A}$.
        + element: $A_{i, j}$.
        + row: $\boldsymbol{A}_{i,:}$.
        + column: $\boldsymbol{A}_{:,j}$.
+ **Tensors**: A tensor is an array with more than two axes.
    + Notations
        + tensor: $\boldsymbol{\mathsf{A}}$
        + element: $\mathsf{A}_{i, j, k}$

The **transpose** of a matrix is the mirror image of the matrix across a diagonal line, called the **main diagonal**, running down and to the right, starting from its upper left corner, $(\boldsymbol{A}^\top)_{i, j} = A_{j, i}$.

We can add matrices to each other, as long as they have the same shape, just by adding their corresponding elements: $\boldsymbol{C} = \boldsymbol{A} + \boldsymbol{B}$, where $C_{i, j} = A_{i, j} + B{i, j}$.

We can also add scalar to a matrix or multiply a matrix by a scalar, just by performing that operation on each element of a matrix: $\boldsymbol{D} = \alpha \cdot \boldsymbol{B} + c$ where $D_{i, j} = \alpha \cdot B_{i, j} + c$.

In the context of deep learning, we also use some less conventional notation. We allow the addition of matrix and a vector, yielding another matrix: $\boldsymbol{C} = \boldsymbol{A} + \boldsymbol{b}$, where $C_{i, j} = A_{i, j} + b_{j}$. In other words, the vector $\boldsymbol{b}$ is added to each row of the matrix. The shorthand eliminates the need to define a matrix with $\boldsymbol{b}$ copied into each row before doing the addition. The implicit copying of $\boldsymbol{b}$ to many locations is called **broadcasting**.

## 2.2 Multiplying Matrices and Vectors
The **matrix product** of matrices $\boldsymbol{A}$ and $\boldsymbol{B}$ is a third matrix $\boldsymbol{C}$. In order for this product to be defined, $\boldsymbol{A}$ must have the same number of columns as $\boldsymbol{B}$ has rows. If $\boldsymbol{A}$ is of shape $m \times n$ and $\boldsymbol{B}$ is $n \times p$, then $\boldsymbol{C}$ is of shape $m \times p$.

The product operation is defined by
$$
C_{i,j} = \sum_{k}{A_{i,k}B_{k,j}}.
$$

The product of the individual elements is called the **element-wise product** or **Hadamard product**, and is denoted as $\boldsymbol{A} \odot \boldsymbol{B}$.

The **dot product** between two vectors $\boldsymbol{x}$ and $\boldsymbol{y}$ of the same dimensionality is the matrix product $\boldsymbol{x}^\top\boldsymbol{y}$.

Properties:

+ Matrix multiplication is distributive:$$\boldsymbol{A}(\boldsymbol{B} + \boldsymbol{C}) = \boldsymbol{AB} + \boldsymbol{AC}.$$
+ Matrix multiplication is associative: $$\boldsymbol{A}(\boldsymbol{BC}) = (\boldsymbol{AB})\boldsymbol{C}.$$
+ Matrix multiplication is *not* commutative, however the dot product between two vectors is commutative:$$\boldsymbol{x}^\top\boldsymbol{y} = \boldsymbol{y}^\top\boldsymbol{x}.$$
+ The transpose of a matrix product has a simple form:$$(\boldsymbol{AB})^\top = \boldsymbol{B}^\top\boldsymbol{A}^\top.$$

## 2.3 Identity and Inverse Matrices
An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix. We denote the identity matrix that preserves $n$-dimensional vectors as $\boldsymbol{I}_n$.

This is $\boldsymbol{I}_3$:

$$
        \begin{bmatrix}
        1 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1 \\
        \end{bmatrix}
$$

The **matrix inverse** of $\boldsymbol{A}$ is denoted as $\boldsymbol{A}^{-1}$, and it is defined as the matrix such that $$\boldsymbol{A}^{-1}\boldsymbol{A} = \boldsymbol{I}_{n}.$$

$\boldsymbol{A}^{-1}$ is primarily useful as a theoretical tool, and should not actually be used in practice for most software applications. Because $\boldsymbol{A}^{-1}$ can be represented with only limited precision on a digital computer, algorithms that make use of the value of $\boldsymbol{b}$ can usually obtain more accurate estimates of $\boldsymbol{x}$.

## 2.4 Linear Dependence and Span
In order for $\boldsymbol{A}^{-1}$ to exist, equation
$$
\boldsymbol{Ax}=\boldsymbol{b}
\tag{1}\label{1}
$$
must have exactly one solution for every value of $\boldsymbol{b}$. However, it is also possible for the system of equations to have no solutions or infinitely many solutions for some value of $\boldsymbol{b}$. It is not possible to have more than one but less than infinitely many solutions for a particular $\boldsymbol{b}$.

A **linear combination** of some set of vectors $\{\boldsymbol{v}^{(1)},\dots,\boldsymbol{v}^{(n)}\}$ is given by multiplying each vector $\boldsymbol{v}^{(i)}$ by a corresponding scalar coefficient and adding the result:
$$
\sum_{i}{c_i\boldsymbol{v}^{(i)}}
$$

The **span** of a set of vectors is the set of all points obtainable by linear combination of the original vectors.

Determining whether $\boldsymbol{Ax} = \boldsymbol{b}$ has a solution thus amounts to testing whether $\boldsymbol{b}$ is in the span of the columns of $\boldsymbol{A}$. This particular span is known as the **column space** or the **range** of $\boldsymbol{A}$.

In order for the system \eqref{1} to have a solution for all values of $\boldsymbol{b} \in \mathbb{R}^{m}$, we therefore require that the column space of $\boldsymbol{A}$ be all of $\mathbb{R}^m$. The requirement implies immediately that $\boldsymbol{A}$ must have at least $m$ columns, i.e., $n \geq m$.

Having $n \geq m$ is only a necessary condition for every point to have a solution. It is not a sufficient condition, because it is possible for the columns to re redundant. Consider a $2 \times 2$ matrix where both of the columns are identical. This has the same column space as a $2 \times 1$ matrix containing only one copy of the replicated column.

This kind of redundancy is known as **linear dependence**. A set of vectors is **linear independent** if no vector in the set is a linear combination of the other vectors.

In order for the matrix to have an inverse, we need to ensure that eqution \eqref{1} has *at most* one solution for each value of $\boldsymbol{b}$. To do so, we need to ensure that the matrix has at most $m$ columns. This means that the matrix must be **square**, that is, we require that $m = n$ and that all of the columns must be linearly independent. A square matrix with linearly dependent columns is known as **singular**.

It is also possible to define an inverse that is multiplied on the right:
$$
\boldsymbol{A}\boldsymbol{A}^{-1} = \boldsymbol{I}.
$$

For square matrices, the left inverse and right inverse are equal.

## 2.5 Norms
In machine learning, we usually measure the size of vectors using a function called a **norm**. Norms are functions  mapping vectors to non-negative values.

The $L^p$ norm is given by
$$
\|\boldsymbol{x}\|_{p} = \left(\sum_{i}{|x_i|^{p}}\right)^{\frac{1}{p}}
$$

for $p \in \mathbb{R}$, $p \geq 1$.

The $L^2$ norm, with $p=2$, is known as the **Euclidean norm**. The $L^2$ norm is used so frequently in machine learning that it is often denoted simply as $\|x\|$, with subscript $2$ omitted. It is also common to measure the size of a vector using the squared $L^2$ norm, which can be calculated simply as $\boldsymbol{x}^\top\boldsymbol{x}$.

The $L^1$ norm may be simplified to
$$
\|x\|_1 = \sum_{i}{|x_i|}
$$

The $L^1$ norm is commonly used in machine learning when the difference between zero and nonzero elements is very important. Every time an element of $\boldsymbol{x}$ moves away from $0$ by $\epsilon$, the $L^1$ norm increases by $\epsilon$.

We sometimes measure the size of the vector by counting its number of nonzero elements. Some authors refer to this function as the "$L^0$ norm", but this is incorrect terminology. The $L^1$ norm is often used as a substitute for the number of nonzero entries.

One other norm that commonly arises in machine learning is the $L^{\infty}$ norm, also known as the **max norm**:

$$
\|x\|_{\infty} = \max_{i}|x_i|.
$$

Sometimes we may also wish to measure the size of a matrix. In the context of deep learning, the most common way to do this is with the otherwise obscure **Frobenius norm**:
$$
\|A\|_{F} = \sqrt{\sum_{i, j}{A_{i, j}^2}}
$$

which is analogous to the $L^2$ norm of a vector.

The dot product of two vectors can be rewritten in terms of norms:
$$
\boldsymbol{x}^\top\boldsymbol{y} = \|x\|_2\,\|y\|_2\,cos\theta
$$
where $\theta$ is the angle between $\boldsymbol{x}$ and $\boldsymbol{y}$.

## 2.6 Special Kinds of Matrices and Vectors
A matrix $\boldsymbol{D}$ is **diagonal** if and only if $D_{i, j}=0$ for all $i \neq j$. We write $\text{diag}(\boldsymbol{v})$ to denote a square diagonal matrix whose diagonal entries are given by the entries of the vector $\boldsymbol{v}$.

Diagonal matrices are of interest in part because multiplying by a diagonal matrix is very computationally efficient. To compute $\text{diag}(\boldsymbol{v})\boldsymbol{x}$, we only need to scale each element $x_i$ by $v_i$. In other words, $\text{diag}(\boldsymbol{v})\boldsymbol{x} = \boldsymbol{v} \odot \boldsymbol{x}$. Inverting a square diagonal matrix is also efficient. The inverse exists only if every diagonal entry is nonzero, and in that case, $\text{diag}(\boldsymbol{v})^{-1} = \text{diag}([1/\boldsymbol{v}_1, \dots, 1/\boldsymbol{v}_n]^\top)$.

A **symmetric** matrix is any matrix that is equal to its own transpose:
$$
A = A^\top.
$$

A **unit vector** is a vector with **unit norm**:
$$
\|x\|_{2} = 1.
$$

A vector $\boldsymbol{x}$ and a vector $\boldsymbol{y}$ are **orthogonal** to each other if $\boldsymbol{x}^\top\boldsymbol{y} = 0$. If both vectors have nonzero norm, this means that they are at a $90$ degree angle to each other. In $\mathbb{R}^n$, at most $n$ vectors may be mutually orthogonal with nonzero norm. If the vectors are not only orthogonal but also have unit norm, we call them **orthonormal**.

An **orthogonal matrix** is a square matrix whose rows are mutually *orthonormal* and whose columns are mutually orthonormal:

$$
\boldsymbol{A}^\top\boldsymbol{A} = \boldsymbol{A}\boldsymbol{A}^\top = \boldsymbol{I}.
$$

This implies that
$$
A^{-1} = A^\top
$$

so orthogonal matrices are of interest because their inverse is very to compute.

## 2.7 Eigendecomposition
In **eigendecomposition**, we decompose a matrix into a set of *eigenvectors* and *eigenvalues*.

An **eigenvector** of a square matrix $\boldsymbol{A}$ is a non-zero vector $\boldsymbol{v}$ such that multiplication by $\boldsymbol{A}$ alters only the scale of $\boldsymbol{v}$:
$$
\boldsymbol{Av} = \lambda \boldsymbol{v}.
$$

The scalar $\lambda$ is known as the **eigenvalue** corresponding to this eigenvector. (One can also find a **left eigenvector** such that $\boldsymbol{v}^\top\boldsymbol{A} = \lambda \boldsymbol{v}^\top$).

If $\boldsymbol{v}$ is an eigenvector of $\boldsymbol{A}$, then so is any rescaled vector $s\boldsymbol{v}$ for $s \in \mathbb{R}$, $s \neq 0$. Moreover, $s\boldsymbol{v}$ still has the same eigenvalue. For this reason, we usually only look for *unit eigenvectors*.

Suppose that a matrix $\boldsymbol{A}$ has $n$ linearly independent eigenvectors, $\{\boldsymbol{v}^{(1)}, \dots, \boldsymbol{v}^{(n)}\}$, with corresponding eigenvalues $\{\lambda_1, \dots, \lambda_n\}$. We may concatenate all of the eigenvectors to form a matrix $\boldsymbol{V}$ with one eigenvector per column: $\boldsymbol{V} = [\boldsymbol{v}^{(1)}, \dots, \boldsymbol{v}^{(n)}]$. Likewise, we can concatenate the eigenvalues to form a vector $\boldsymbol{\lambda} = [\lambda_1, \dots, \lambda_n]^\top$. The **eigendecomposition** of $\boldsymbol{A}$ is then given by
$$
\boldsymbol{A = \boldsymbol{V}\text{diag}(\boldsymbol{\lambda})\boldsymbol{V}^{-1}}
$$

Not every matrix can be decomposed into eigenvalues and eigenvectors. In some cases, the decomposition exists, but may involve *complex* rather than real numbers. Specifically, every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues:
$$
\boldsymbol{A} = \boldsymbol{Q \Lambda Q^\top}
$$
where $\boldsymbol{Q}$ is an orthogonal matrix composed of eigenvectors of \boldsymbol{A}, and $\boldsymbol{\Lambda}$ is a diagonal matrix. The eigenvalue $\Lambda_{i, i}$ is associated with the eigenvector in the column $i$ of $\boldsymbol{Q}$, denoted as $\boldsymbol{Q}_{:,i}$. Because $\boldsymbol{Q}$ is an orthogonal matrix, we can think of $\boldsymbol{A}$ as scaling space by $\lambda_{i}$ in direction $\boldsymbol{v}^{(i)}$.

While any real symmetric matrix $\boldsymbol{A}$ is guaranteed to have an eigendecomposition, the eigendecomposition may not be unique. If any two or more eigenvectors share the same eigenvalue, then any set of orthogonal vectors lying in their span are also eigenvectors with that eigenvalue. By convention, we usually sort the entries of $\boldsymbol{\Lambda}$ in descending order. Under this convention, the eigendecomposition is unique only if all of the eigenvalues are unique.

The matrix is singular if and only if any of the eigenvalues are zero.

A matrix whose eigenvalues are all positive is called **positive definite**. A matrix whose eigenvalues are all positive or zero-values is called **positive semi-definite**. Likewise, if all eigenvalues are negative, the matrix is called **negative definite**, and if all eigenvalues are negative or zero-valued, it is **negative-semidefinite**. Positive semidefinite matrices are interesting because they guarantee that $\forall \boldsymbol{x}$, $\boldsymbol{x}^{\top}\boldsymbol{Ax} \geq0$

## 2.8 Singular Value Decomposition
The singular value decomposition (SVD) provides another way to factorize a matrix, into **singular vectors** and **singular values**. Every real matrix has a singular value decomposition, but the same is not true of the eigenvalue decomposition. For example, if a matrix is not square, the eigendecomposition is not defined, and we must use a singular value decomposition instead.

In singular value decomposition, we will write $\boldsymbol{A}$ as a product of three matrices:
$$
\boldsymbol{A} = \boldsymbol{UDV^\top}
$$

Suppose that $\boldsymbol{A}$ is an $m \times n$ matrix. Then $\boldsymbol{U}$ is defined to be an $m \times m$ matrix, $\boldsymbol{D}$ to be an $m \times n$ matrix, and $\boldsymbol{V}$ to be an $n \times n$ matrix.

The matrices $\boldsymbol{U}$ and $\boldsymbol{V}$ are both defined to be *orthogonal* matrices. The matrix $\boldsymbol{D}$ is defined to be a *diagonal* matrix. Note that $\boldsymbol{D}$ is not necessarily square.

The elements along the diagonal of $\boldsymbol{D}$ are known as the **singular values** of the matrix $\boldsymbol{A}$. The columns of $\boldsymbol{U}$ are known as the **left-singular vectors**. The columns of $\boldsymbol{V}$ are known as the **right-singular vectors**.

We can actually interpret the singular value decomposition of $\boldsymbol{A}$ in terms of the eigendecomposition of functions of $\boldsymbol{A}$. The left-singular vectors of $\boldsymbol{A}$ are the eigenvectors of $\boldsymbol{AA^\top}$. The right-singular vectors of $\boldsymbol{A}$ are the eigenvectors of $\boldsymbol{A^\top}A$. The non-zero singular values of $\boldsymbol{A}$ are the square roots of the eigenvalues of $\boldsymbol{A^{\top}A}$. The same is true for $\boldsymbol{A A^\top}$.

## 2.9 The Moore-Penrose Pseudoinverse
Matrix inversion is not defined for matrices that are not square. Suppose we want to make a left-inverse $\boldsymbol{B}$ of $\boldsymbol{A}$, so that we can solve a linear equation
$$
\boldsymbol{Ax} = \boldsymbol{y}
$$
by left-multiplying each side to obtain
$$
\boldsymbol{x}=\boldsymbol{By}
$$

Depending on the structure of the problem, it may not be possible to design a unique mapping from $\boldsymbol{A}$ to $\boldsymbol{B}$.

If $\boldsymbol{A}$ is taller than it is wide, then it is possible for this equation to have no solution. If $\boldsymbol{A}$ is wider than it is tall, then there could be multiple possible solutions.

The **Moore-Penrose pseudoinverse** allow us to make some headway in these cases. The pseudoinverse of $\boldsymbol{A}$ is defined as a matrix
$$
\boldsymbol{A}^{+} = \lim_{\alpha \searrow 0}(\boldsymbol{A}^{\top}\boldsymbol{A} + \alpha\boldsymbol{I})^{-1}\boldsymbol{A}^{\top}
$$

Practical algorithms for computing the pseudoinverse are not based on this definition, but rather the formula

$$
\boldsymbol{A}^{+} = \boldsymbol{VD^{+}U^\top}
$$

where $\boldsymbol{U}$, $\boldsymbol{D}$ and $\boldsymbol{V}$ are the singular value decomposition of $\boldsymbol{A}$, and the pseudoinverse $\boldsymbol{D^+}$ of a diagonal matrix $\boldsymbol{D}$ is obtained by taking the reciprocal of its non-zero elements then taking the transpose of the resulting matrix.

When $\boldsymbol{A}$ has more columns than rows, then solving a linear equation using the pseudoinverse provides one of the many possible solutions. Specifically, it provides the solution $\boldsymbol{x} = \boldsymbol{A}^+\boldsymbol{y}$ with minimal Euclidean norm $\|\boldsymbol{x}^2\|$ among all possible solutions.

When $\boldsymbol{A}$ has more rows than columns, it is possible for there to be no solution. In this case, using the pseudoinverse gives us the $\boldsymbol{x}$ for which $\boldsymbol{Ax}$ is as close as possible to $\boldsymbol{y}$ in terms of Euclidean norm $\|\boldsymbol{Ax} - \boldsymbol{y}\|$.

## 2.10 The Trace Operator
The trace operator gives the sum of all of the diagonal entries of a matrix

$$
\text{Tr}(\boldsymbol{A}) = \sum_{i}{\boldsymbol{A}_{i, i}}
$$

The trace operator provides an alternative way of writing the Frobenius norm of a matrix:
$$
\|\boldsymbol{A}\|_{F} = \sqrt{\text{Tr}(\boldsymbol{AA}^{\top})}
$$

The trace operator is invariant to the transpose operator:
$$
\text{Tr}(\boldsymbol{A}) = \text{Tr}(\boldsymbol{A}^\top)
$$

The trace of a matrix compound of many factors is also invariant to moving the last factor into the first position, if the shapes of the corresponding matrices allow the resulting product to be defined:
$$
\text{Tr}(\boldsymbol{ABC}) = \text{Tr}(\boldsymbol{CAB}) = \text{Tr}(\boldsymbol{BCA})
$$

or more generally,
$$
\text{Tr}\left(\prod^{n}_{i=1}\boldsymbol{F}^{(i)}\right) = \text{Tr}\left(\boldsymbol{F}^{(n)}\prod_{i=1}^{n-1}\boldsymbol{F}^{(i)}\right)
$$

This invariance to cyclic permutation holds even if the resulting product has a different shape. For example, for $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ and $\boldsymbol{B} \in \mathbb{R}^{n \times m}$, we have
$$
\text{Tr}(\boldsymbol{AB}) = \text{Tr}(\boldsymbol{BA})
$$
even though $\boldsymbol{AB} \in \mathbb{R}^{m \times m}$ and $\boldsymbol{BA} \in \mathbb{R}^{n \times n}$.

A scalar is its own trace:
$$
\alpha = \text{Tr}(\alpha)
$$

## 2.11 The Determinant
The determinant of a square matrix, denoted $\text{det}(\boldsymbol{A})$, is a function mapping matrices to real scalars. The determinant is equal to the product of all the eigenvalues of the matrix. The absolute value of the determinant can be thought of as a measure of how much multiplication by the matrix expands or contracts space. If the determinant is $0$, then space is contracted completely along at least one dimension, causing it to lose all of its volume. If the determinant is $1$, then the transformation preserves volume.

## 2.12 Example: Principal Components Analysis
One simple machine learning algorithm, **principal compinents analysis** or PCA can be derived using only knowledge of basic linear algebra.

Suppose we have a collection of $m$ points $\{\boldsymbol{x^{(1)}},\dots,\boldsymbol{x^{(m)}}\}$ in $\mathbb{R}^{n}$. Suppose we would like to apply lossy compression to these points. Lossy compression means storing the points in a way that requires less memory but may lose some precision. We would like to lose as little precision as possible.

One way we can encode these points is to represent a lower-dimensional version of them. For each point $\boldsymbol{x^{(i)}} \in \mathbb{R}^n$ we will find a corresponding code vector $\boldsymbol{c}^{(i)} \in \mathbb{R}^{l}$. If $l$ is smaller than $n$, it will take less memory to store the code points than the original data. We will want to find some *encoding function* that produces the code for an input, $f(\boldsymbol{x}) = \boldsymbol{c}$, and a *decoding function* that produces the reconstructed input given its code, $\boldsymbol{x} \approx g(f(\boldsymbol{x})).$

PCA is defined by our choice of the decoding function. Specifically, to make the decoder very simple, we choose to use matrix multiplication to map the code back into $\mathbb{R}^n$. Let $g(\boldsymbol{c}) = \boldsymbol{Dc}$, where $\boldsymbol{D} \in \mathbb{R}^{n \times l}$ is the matrix defining the decoding.

Computing the optimal code for this decoder could be a difficult problem. To keep the encoding problem easy, PCA constrains the columns of $\boldsymbol{D}$ to be orthogonal to each other.

With the problem as described so far, many solutions are possible, because we can increase the scale of $\boldsymbol{D}_{:,i}$ if we decrease $c_i$ proportionally for all points. To give the problem a unique solution, we constrain all of the columns of $\boldsymbol{D}$ to have unit norm.

In order to turn this basic idea into an algorithm we can implement, the first thing we need to do is figure out how to generate the optimal code point $\boldsymbol{c}^{*}$ for each input point $\boldsymbol{x}$. One way to do this is to minimize the distance between the input point $\boldsymbol{x}$ and its reconstruction, $g(\boldsymbol{c}^{*})$. We can measure this distance using a norm. In the principal components algorithm, we use the $L^2$ norm:
$$
\boldsymbol{c}^* = \arg\,\min_{\boldsymbol{c}}\|\boldsymbol{x} - g(\boldsymbol{c})\|_2
$$

We can switch to the squared $L^2$ norm instead of the $L^2$ norm itself, because both are minimized by the same value of $\boldsymbol{c}$. Both are minimized by the same value of $\boldsymbol{c}$ because the $L^2$ norm is non-negative and the squaring operation is monotonically increasing for non-negative arguments.
$$
\boldsymbol{c}^* = \arg\,\min_{\boldsymbol{c}}\|\boldsymbol{x}-g(\boldsymbol{c})\|_{2}^{2}
$$

The function being minimized simplifies to 
$$
(\boldsymbol{x} - g(\boldsymbol{c}))^\top(\boldsymbol{x}-g(\boldsymbol{c}))
$$
(by the definition of the $L^2$ norm)

$$
=\boldsymbol{x}^\top\boldsymbol{x} - \boldsymbol{x}^\top g(\boldsymbol{c}) - g(\boldsymbol{c})^\top\boldsymbol{x} + g(\boldsymbol{c})^\top g(\boldsymbol{c})
$$
(by the distributive property)

$$
=\boldsymbol{x}^\top\boldsymbol{x} - 2\boldsymbol{x}^\top g(\boldsymbol{c}) + g(\boldsymbol{c})^\top g(\boldsymbol{c})
$$

(because the scalar $g(\boldsymbol{c})^\top \boldsymbol{x}$ is equal to the transpose of itself)

We can now change the function being minimized again, to omit the first term, since this term does not depend on $\boldsymbol{c}$:

$$
\boldsymbol{c}^* = \arg\,\min_{\boldsymbol{c}} -2\boldsymbol{x}^\top g(\boldsymbol{c}) + g(\boldsymbol{c})^\top g(\boldsymbol{c})
$$

To make further progress, we must substitute in the definition of $g(\boldsymbol{\boldsymbol{c}})$:

$$
\begin{align*}
\boldsymbol{c}^* & = \arg\,\min_{\boldsymbol{c}} -2\boldsymbol{x}^\top \boldsymbol{Dc} + \boldsymbol{c}^\top \boldsymbol{D}^\top\boldsymbol{Dc}\\
 & = \arg\,\min_{\boldsymbol{c}} -2\boldsymbol{x}^\top \boldsymbol{Dc} + \boldsymbol{c}^\top \boldsymbol{I}_{l}\boldsymbol{c}\\ 
\end{align*}$$

(by the orthogonality and unit norm constraints on \boldsymbol{D})

$$
\begin{align*}
 & = \arg\,\min_{\boldsymbol{c}} -2\boldsymbol{x}^\top \boldsymbol{Dc} + \boldsymbol{c}^\top\boldsymbol{c}\\ 
\end{align*}$$

We can solve this optimization problem using vector calculus:

$$
\nabla_{\boldsymbol{c}}(-2\boldsymbol{x}^\top\boldsymbol{Dc} + \boldsymbol{c}^\top\boldsymbol{c}) = \boldsymbol{0}
$$

$$
-2\boldsymbol{D}^\top\boldsymbol{x} + 2\boldsymbol{c} = \boldsymbol{0}
$$

$$
\boldsymbol{c} = \boldsymbol{D}^\top \boldsymbol{x}
$$

This makes the algorithm efficient: we can optimally encode $\boldsymbol{x}$ just using a matrix-vector operation. To encode a vector, we apply the encoder function

$$
f(\boldsymbol{x}) = \boldsymbol{D}^\top\boldsymbol{x}
$$

Using a further matrix multiplication, we can also define the PCA reconstruction operation:

$$
\tag{2}\label{2}
r(\boldsymbol{x}) = g(f(x)) = \boldsymbol{DD^{\top}x}
$$

Next, we need to choose the encoding matrix $\boldsymbol{D}$. To do so, we revisit the ideas of minimizing the $L^2$ distance between inputs and reconstructions. Since we will use the same matrix $\boldsymbol{D}$ to decode all of the points, we can no longer consider the points in isolation. Instead, we must minimize the Frobenius norm of the matrix of errors computed over all dimensions and all points.

$$\tag{3}\label{3}
\boldsymbol{D}^* = \arg\,\min_{\boldsymbol{D}} \sqrt{\sum_{i,j}\left(x_{j}^{(i)}-r(\boldsymbol{x^{(i)}})_j\right)^2} \text{subject to } \boldsymbol{D^\top D} = \boldsymbol{I}_{l}
$$

To derive the algorithm for finding $\boldsymbol{D}^*$, we will start by considering the case where $l=1$. In this case, $\boldsymbol{D}$ is just a single vector $\boldsymbol{d}$. Substituting equation $\eqref{2}$ into $\eqref{3}$ and simplifying $\boldsymbol{D}$ into $\boldsymbol{d}$, the problem reduces to

$$
d^* = \arg\,\min_{\boldsymbol{d}} \sum_{i}\|\boldsymbol{x}^{(i)}-\boldsymbol{dd^\top x^{(i)}}\|_{2}^{2} \;\; \text{ subject to } \|\boldsymbol{d}\|_2=1,
$$

or exploiting the fact that a scalar is its own transpose, as

$$
d^* = \arg\,\min_{\boldsymbol{d}} \sum_{i}\|\boldsymbol{x}^{(i)}- \boldsymbol{x}^{(i)} \boldsymbol{dd^\top }\|_{2}^{2} \;\; \text{ subject to } \|\boldsymbol{d}\|_2=1,
$$

At this point, it can be helpful to rewrite the problem in terms of a single design matrix of examples, rather than as a sum over separate example vectors. This allow us to use more compact notation. Let $\boldsymbol{X} \in \mathbb{R}^{m \times n}$ be the matrix defined by stacking all of the vectors describing the points, such that $\boldsymbol{X}_{i,:} = \boldsymbol{x}^{(i)\top}$.
We can now rewrite the problem as

$$
\boldsymbol{d}^* = \arg\,\min_{\boldsymbol{d}} \|\boldsymbol{X}-\boldsymbol{Xdd^\top}\|_{F}^{2}\;\;\text{ subject to } \boldsymbol{d^\top d} = 1
$$

Disregarding the constraint for the moment, we can simplify the Frobenius norm portion as follows:

$$
\arg\,\min_{\boldsymbol{d}} \|\boldsymbol{X}-\boldsymbol{Xdd^\top}\|^{2}_{F}
$$

$$
= \arg\,\min_{\boldsymbol{d}} \text{Tr}\left((\boldsymbol{X}-\boldsymbol{Xdd^\top})^\top(\boldsymbol{X}-\boldsymbol{Xdd^\top})\right)
$$

$$
=\arg\,\min_{\boldsymbol{d}} \text{Tr}(\boldsymbol{X^\top X}-\boldsymbol{X^\top Xdd^\top}-\boldsymbol{dd^\top X^\top X} + \boldsymbol{dd^\top X^\top Xdd^\top})
$$

$$
=\arg\,\min_{\boldsymbol{d}} \text{Tr}(\boldsymbol{X^\top X}) -\text{Tr}(\boldsymbol{X^\top Xdd^\top}) - \text{Tr}(\boldsymbol{dd^\top X^\top X}) + \text{Tr}(\boldsymbol{dd^\top X^\top Xdd^\top})
$$

$$
=\arg\,\min_{\boldsymbol{d}} -\text{Tr}(\boldsymbol{X^\top Xdd^\top}) - \text{Tr}(\boldsymbol{dd^\top X^\top X}) + \text{Tr}(\boldsymbol{dd^\top X^\top Xdd^\top})
$$

(because terms not involving $\boldsymbol{d}$ do not affect the $\arg\,\min$)

$$
=\arg\,\min_{\boldsymbol{d}} -2\text{Tr}(\boldsymbol{X^\top Xdd^\top}) + \text{Tr}(\boldsymbol{dd^\top X^\top Xdd^\top})
$$

(because we can cycle the order of the matrices inside a trace)

$$
=\arg\,\min_{\boldsymbol{d}} -2\text{Tr}(\boldsymbol{X^\top Xdd^\top}) + \text{Tr}(\boldsymbol{X^\top Xdd^\top dd^\top})
$$

(as above)

At this point, we re-introduce the constraint:

$$
\arg\,\min_{\boldsymbol{d}} -2\text{Tr}(\boldsymbol{X^\top Xdd^\top}) + \text{Tr}(\boldsymbol{X^\top Xdd^\top dd^\top}) \;\; \text{ subject to } \boldsymbol{d^\top d =1}
$$

$$
\arg\,\min_{\boldsymbol{d}} -2\text{Tr}(\boldsymbol{X^\top Xdd^\top}) + \text{Tr}(\boldsymbol{X^\top Xdd^\top}) \;\; \text{ subject to } \boldsymbol{d^\top d =1}
$$

(due to the constraint)

$$
\arg\,\min_{\boldsymbol{d}} -\text{Tr}(\boldsymbol{X^\top Xdd^\top}) \;\; \text{ subject to } \boldsymbol{d^\top d =1}
$$

$$
= \arg\,\max_{\boldsymbol{d}} -\text{Tr}(\boldsymbol{X^\top Xdd^\top}) \;\; \text{ subject to } \boldsymbol{d^\top d =1}
$$

$$
= \arg\,\max_{\boldsymbol{d}} -\text{Tr}(\boldsymbol{d^\top X^\top Xd}) \;\; \text{ subject to } \boldsymbol{d^\top d =1}
$$

This optimization problem may be solved using eigendecomposition. Specifically, the optimal $\boldsymbol{d}$ is given by the eigenvector of $\boldsymbol{X}^\top \boldsymbol{X}$ corresponding to the largest eigenvalue.

This derivation is specific to the case of $l=1$ and recovers only the first  principal components, the matrix $\boldsymbol{D}$ is given by the $l$ eigenvectors corresponding to the largest eigenvalues. This is may be shown using proof by induction.

(END OF CHAPTER 2)


