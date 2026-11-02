# Learning with Augmented Features for Supervised and Semi-Supervised Heterogeneous Domain Adaptation

Wen Li, Lixin Duan, Dong Xu, and Ivor W. Tsang. IEEE TPAMI, 36(6):1134–1148, 2014.
DOI: 10.1109/TPAMI.2013.167.

This note restates the algorithm in the form that is actually optimized, and it lines that form up with the authors' MATLAB release (`matlab_implementation/official_shfa_release/`, from [wenli-vision/SHFA_release](https://github.com/wenli-vision/SHFA_release)). The file `python_implementation/hfa_numpy.py` is not this algorithm. It is an alternating linear hinge solver of the kind the paper explicitly replaces, because that alternation is not known to converge to the global solution. The faithful Python port is `python_implementation/reproduce_official.py`.

## 1. Problem

There is one source domain and one target domain. The features do not live in the same space.

- Source sample \(x^s_i \in \mathbb{R}^{d_s}\) with label \(y^s_i \in \{+1,-1\}\), \(i=1,\ldots,n_s\).
- Labeled target sample \(x^t_i \in \mathbb{R}^{d_t}\) with label \(y^t_i \in \{+1,-1\}\), \(i=1,\ldots,n_t\).
- For the semi-supervised method, unlabeled target samples \(x^u_i \in \mathbb{R}^{d_t}\), \(i=1,\ldots,n_u\), whose labels \(y^u_i\) are unknown.

In heterogeneous domain adaptation, \(d_s \neq d_t\). A classifier trained on the source cannot be applied to the target, and simply padding the shorter vector with zeros does not create a correspondence between coordinates. The standard homogeneous trick of feature replication (Daumé, 2007),

\[
\phi_s(x)=[x^\top, x^\top, 0^\top]^\top, \qquad \phi_t(x)=[x^\top, 0^\top, x^\top]^\top,
\]

has the same problem: the two copies of \(x\) only make sense when the coordinates already mean the same thing.

Multi-class problems are reduced to one-versus-all binary problems. Each category gets its own augmented-feature model. At test time the category with the largest decision value is chosen.

## 2. Augmented features

Choose a common dimension \(d_c\) and two projections \(P \in \mathbb{R}^{d_c \times d_s}\), \(Q \in \mathbb{R}^{d_c \times d_t}\). Map a source point and a target point by

\[
\varphi_s(x^s)=
\begin{bmatrix} P x^s \\ x^s \\ 0_{d_t} \end{bmatrix},
\qquad
\varphi_t(x^t)=
\begin{bmatrix} Q x^t \\ 0_{d_s} \\ x^t \end{bmatrix}.
\tag{1}
\]

The first block is where the two domains become comparable. The second and third blocks keep the original features of each domain, padded by zeros so that both mapped vectors have length \(d_c+d_s+d_t\). Samples from the same domain can still match through their original features. Samples from different domains can match only through \(P x^s\) and \(Q x^t\).

Any linear supervised learner can now be trained on these vectors. The paper uses SVM as the showcase. The projections are not chosen in advance. They are learned together with the classifier.

## 3. Supervised HFA

### 3.1 Primal

Split the augmented weight as \(w=[w_c^\top, w_s^\top, w_t^\top]^\top\). HFA solves

\[
\min_{P,Q}\ \min_{w,b,\xi}\ 
\frac12\|w\|^2
+ C\sum_{i=1}^{n_s}\xi^s_i
+ C\sum_{i=1}^{n_t}\xi^t_i
\tag{2}
\]

subject to the usual hinge constraints

\[
\begin{aligned}
y^s_i\big(w^\top\varphi_s(x^s_i)+b\big) &\ge 1-\xi^s_i, & \xi^s_i\ge 0,\\
y^t_i\big(w^\top\varphi_t(x^t_i)+b\big) &\ge 1-\xi^t_i, & \xi^t_i\ge 0,
\end{aligned}
\]

and the Frobenius ball constraints \(\|P\|_F^2\le \lambda_P\), \(\|Q\|_F^2\le \lambda_Q\). The release uses a common box constraint \(C\) on every training point (\(C_s=C_t=1\) in `demo.m`). Different \(C_s\) and \(C_t\) are implemented as LIBSVM instance weights.

### 3.2 Dual in \(P\) and \(Q\)

Let \(\alpha\) be the vector of all dual hinge multipliers, and let \(y\) be the concatenated \(\pm 1\) label vector. The inner SVM dual, still depending on \(P\) and \(Q\), is

\[
\min_{P,Q}\ \max_{\alpha\in\mathcal{A}}\ 
\mathbf{1}^\top\alpha - \tfrac12 (\alpha\circ y)^\top K_{P,Q}(\alpha\circ y),
\tag{5}
\]

where \(\mathcal{A}=\{\alpha: y^\top\alpha=0,\ 0\le\alpha\le C\mathbf{1}\}\) and the Gram matrix of the augmented features is block-structured:

\[
K_{P,Q}=
\begin{bmatrix}
X_s^\top(I+P^\top P)X_s & X_s^\top P^\top Q X_t \\
X_t^\top Q^\top P X_s & X_t^\top(I+Q^\top Q)X_t
\end{bmatrix}.
\]

\(P\) and \(Q\) appear only as the products \(P^\top P\), \(P^\top Q\), \(Q^\top P\), and \(Q^\top Q\). Define the transformation metric

\[
H=\begin{bmatrix}P\\ Q\end{bmatrix}\begin{bmatrix}P^\top & Q^\top\end{bmatrix}
\in\mathbb{R}^{(d_s+d_t)\times(d_s+d_t)},
\qquad H\succeq 0.
\]

With the block-diagonal data matrix \(X=\mathrm{blkdiag}(X_s,X_t)\) and \(\lambda=\lambda_P+\lambda_Q\),

\[
\min_{H\succeq 0}\ \max_{\alpha\in\mathcal{A}}\ 
\mathbf{1}^\top\alpha - \tfrac12 (\alpha\circ y)^\top X^\top(H+I)X\,(\alpha\circ y)
\quad\text{s.t.}\quad \mathrm{tr}(H)\le\lambda.
\tag{6}
\]

Two practical consequences:

- The common dimension \(d_c\) disappears. The common subspace is latent.
- \(H\) is still linear in the original features, and its size grows with \(d_s+d_t\).

### 3.3 Kernelization

Let \(K_s\) and \(K_t\) be kernel matrices on the source points and the target points, with feature maps \(\phi_s\) and \(\phi_t\). The same derivation applies in those feature spaces. The explicit maps are unknown and may be infinite-dimensional, so the paper changes variables. If

\[
K=\mathrm{blkdiag}(K_s,K_t)
\]

and \(K^{1/2}\) is its symmetric square root, there is a matrix \(\tilde H\succeq 0\) of size \((n_s+n_t)\times(n_s+n_t)\) such that the augmented kernel is

\[
K_{\tilde H}=K^{1/2}(\tilde H+I)K^{1/2},
\qquad \mathrm{tr}(\tilde H)=\mathrm{tr}(H)\le\lambda.
\]

The optimized object no longer depends on \(d_s\) or \(d_t\). Below, \(H\) means this sample-sized metric. The linear method is the same program with \(K^{1/2}\) replaced by \(X\).

The release builds the Gaussian kernel used in the paper's experiments. For column-normalized points, `return_GaussianKernel.m` sets

\[
D_{ij}=\|x_i-x_j\|^2,
\qquad
\sigma=\sqrt{\tfrac12\mathrm{mean}(D)},
\qquad
\gamma=\frac{1}{2\sigma^2},
\qquad
K=\exp(-\gamma D).
\]

The mean includes the zero diagonal. The bandwidth is computed on each domain's own training points and then reused for test kernels. `demo.m` takes the real part of `sqrtm` on \(K_s\) and on \(K_t\) separately, then assembles a block-diagonal square root. That is valid because \(K\) is block diagonal.

### 3.4 Convex reformulation as infinite-kernel learning

The program in \(H\) is still a semi-definite program if it is attacked directly. The preliminary ICML 2012 version alternated an SVM step in \(\alpha\) with an SDP step in \(H\). The TPAMI paper discards that alternation. Trace-norm control can be written as a Tikhonov term \(\eta\,\mathrm{tr}(H)\) for an equivalent penalty \(\eta\). Any feasible \(H\) is a conic combination of normalized rank-one matrices

\[
\mathcal{M}=\{hh^\top: h\in\mathbb{R}^{n},\ \|h\|=1\},
\qquad
H_\theta=\sum_{r=1}^{\infty}\theta_r M_r,
\quad \theta\ge 0,
\quad M_r\in\mathcal{M}.
\]

Because \(\mathrm{tr}(hh^\top)=1\), one has \(\mathrm{tr}(H_\theta)=\mathbf{1}^\top\theta\). Substituting and rescaling \(\theta\) by \(\lambda\) produces a standard multiple-kernel learning (MKL) problem, with one base kernel per rank-one atom:

\[
\min_{\theta\in\mathcal{D}}
\max_{\alpha\in\mathcal{A}}
\mathbf{1}^\top\alpha
-\tfrac12(\alpha\circ y)^\top
\Big(\sum_{r=1}^{\infty}\theta_r K_r\Big)
(\alpha\circ y),
\tag{11}
\]

\[
K_r=K^{1/2}(\lambda h_r h_r^\top+I)K^{1/2},
\qquad
\mathcal{D}=\{\theta:\ \theta\ge 0,\ \mathbf{1}^\top\theta\le 1\}.
\]

This is convex in the MKL sense: the inner maximization is a convex quadratic program, and the outer problem is an infinite-kernel learning problem. A finite number of atoms is enough to represent the optimum. The global solution is obtained by a cutting-plane method rather than by alternating \(P\) and \(Q\).

### 3.5 Cutting plane (Algorithm 1)

Only the most violated rank-one constraint is needed at each round. For a fixed \(\alpha\), the kernel that most violates the dual constraint has the closed form

\[
h=\frac{K^{1/2}(\alpha\circ y)}{\|K^{1/2}(\alpha\circ y)\|},
\qquad
M=hh^\top.
\]

The release absorbs \(\sqrt{\lambda}\) into the stored vector, so the column appended to `Hvs` is \(\sqrt{\lambda}\,h\), and the base kernel is \(K^{1/2}(vv^\top+I)K^{1/2}\) with \(v\) that column. The identity part is shared. With coefficients \(d\) (the finite \(\theta\)),

\[
H=\sum_r d_r v_r v_r^\top,
\qquad
K(d)=K^{1/2}(H+I)K^{1/2}.
\]

`LpMKL_H_fast.m` alternates

1. a C-SVM on the current combined kernel (`-t 4 -c 1`, instance weights equal to \(C_s\) or \(C_t\));
2. the \(p\)-norm MKL weight update. For \(p=1\), which is `demo.m` (`mkl_degree = 1`),

\[
w_r=\|u_r\|,\qquad d \leftarrow \frac{w}{\|w\|_1},
\]

where \(u_r\) is the portion of the weight vector that sits on base kernel \(r\). The SVM dual objective

\[
\mathbf{1}^\top\alpha-\tfrac12(\alpha\circ y)^\top K(d)\,(\alpha\circ y)
\]

is what the outer loop monitors.

**Algorithm 1, as implemented.**

1. Set \(v_1=\sqrt{\lambda/n}\,\mathbf{1}\) and \(r=1\). This is \(\sqrt{\lambda}\) times the paper's uniform unit vector.
2. Solve the \(p\)-norm MKL problem on the current columns to get \(d\) and \(\alpha\).
3. Stop when the outer objective changes by at most \(\tau\) relative to itself, or after 50 outer iterations. The release uses \(\tau=10^{-3}\). The inner MKL loop uses the same tolerance and at most 100 SVM solves.
4. Otherwise append \(v_{r+1}=\sqrt{\lambda}\,K^{1/2}(\alpha\circ y)/\|K^{1/2}(\alpha\circ y)\|\) and repeat.

The objective of this loop decreases monotonically and the finite cutting-plane solution reaches the infinite-kernel optimum when it stabilizes (Theorem 2 of the paper). In the release the outer cap is 50 iterations, which the paper says is enough in practice.

### 3.6 Target decision function

After \(H\) and \(\alpha\) are learned, a target test kernel vector \(k_t\) (similarities to the labeled target training points only) is scored by

\[
f(x)=(\alpha\circ y)^\top K^{1/2}(H+I)
\begin{bmatrix}0\\ K_t^{-1/2}k_t\end{bmatrix}+b.
\tag{14}
\]

`demo.m` implements this as

```matlab
tmp = (K_test * L_t_inv' * H * K_root);
dec = tmp * y_alpha + K_test * y_alpha_t - rho;
```

with \(L_{t}^{-1}=[0;\ K_t^{-1/2}]\) and \(y_\alpha=\alpha\circ y\). The bias sign follows LIBSVM's `rho`, flipped if LIBSVM's first label is \(-1\), so that a positive score supports the positive class. One-versus-all then takes the class with the largest score.

## 4. Semi-supervised HFA (SHFA)

HFA never sees the unlabeled target points. SHFA puts them into the same augmented feature space and infers their labels while learning \(H\).

### 4.1 Primal

The paper switches the loss to the square hinge and uses \(\rho\)-SVM, so the margin \(\rho\) is a variable rather than being fixed at 1:

\[
\min_{y^u,w,b,\rho,P,Q,\xi}\ 
\frac12\|w\|^2+\frac12 b^2-\rho
+\frac{C_u}{2}\sum_{i=1}^{n_u}(\xi^u_i)^2
\tag{15}
\]

subject to

\[
\begin{aligned}
y^s_i(w^\top\varphi_s(x^s_i)+b)&\ge\rho-\xi^s_i,\\
y^t_i(w^\top\varphi_t(x^t_i)+b)&\ge\rho-\xi^t_i,\\
y^u_i(w^\top\varphi_t(x^u_i)+b)&\ge\rho-\xi^u_i,
\end{aligned}
\]

the same bounds on \(P\) and \(Q\), and the balance constraint \(\mathbf{1}^\top y^u=\delta\). The balance constraint is the same device used by transductive SVM: it stops the trivial solution that labels every unlabeled point with the majority class. In the release, \(\delta\) is not passed directly. The code forces the positive fraction of \(y^u\) to the known positive rate of the test labels (`upper_ratio = lower_ratio =` that rate). The paper states that this prior is the ground-truth ratio, and that the same prior is given to the T-SVM baseline. It is class-prior information, not the individual test labels.

The kernel is now built on source points plus labeled and unlabeled target points, still block diagonal across domains. The target block contains both labeled and unlabeled target points. The dual, up to the paper's notation for the quadratic term, has the form

\[
\min_{y\in\mathcal{Y},\,H\succeq 0}\ \max_{\alpha\in\mathcal{A}}\ 
-\tfrac12\alpha^\top(Q_{H,y}+D)\alpha
\quad\text{s.t.}\quad\mathrm{tr}(H)\le\lambda,
\tag{16}
\]

where \(\mathcal{A}=\{\alpha:\alpha\ge 0,\ \mathbf{1}^\top\alpha=1\}\), \(D\) is diagonal with entries \(1/C\) on labeled points and \(1/C_u\) on unlabeled points, and

\[
Q_{H,y}=K^{1/2}(H+I)K^{1/2}+\mathbf{1}\mathbf{1}^\top\circ(yy^\top).
\]

\(\mathcal{Y}\) is the set of label vectors whose source and labeled-target parts are fixed and whose unlabeled part has the prescribed balance.

The released solver does not hand this quadratic form to a generic QP solver. It folds the \(\pm 1\) labels into the kernel by the Schur product with \(yy^\top\) and then calls one-class SVM (`-s 2 -t 4 -n 1/n`). One-class SVM with \(\nu=1/n\) enforces \(0\le\alpha_i\le 1\) and \(\mathbf{1}^\top\alpha=1\). Adding \(D\) to the kernel diagonal inserts the square-hinge term. With the demo parameters \(C_s=C_t=1\) and \(C_u=C_x=10^{-3}\), the diagonal weights are \(1\), \(1\), and \(1000\).

### 4.2 Convex relaxation

Inferring \(y^u\) exactly is a mixed-integer program. The paper lower-bounds that problem by allowing a convex combination of many feasible labelings \(y_l\) with weights \(\gamma\), and it expands \(H\) into rank-one atoms as in HFA. The product of the two sets of weights is again a single coefficient vector \(d\) on pairs \((M_k,y_k)\):

\[
\min_{d\in\mathcal{D}}
\max_{\alpha\in\mathcal{A}}
-\tfrac12\alpha^\top\Big(\sum_k d_k Q_{M_k,y_k}+D\Big)\alpha,
\qquad
\mathbf{1}^\top d\le 1,\ d\ge 0.
\tag{19}
\]

Each pair is one base kernel. The \(p\)-norm extension in Section 3.4 of the paper replaces \(\|d\|_1\le 1\) by \(\|d\|_p\le 1\). The demo sets \(p=1.5\) (`mkl_degree = 1.5`), which the paper reports as the better of the two settings it tried. The MKL update in `LpMKL_H_labels_v4.m` is the standard one:

\[
d_r \propto \|u_r\|^{2/(p+1)},
\qquad
\|d\|_p = 1.
\]

### 4.3 Finding the next labeling (Algorithm 2)

There are infinitely many pairs \((M,y)\). The most violated pair can be reduced, after substituting the optimal rank-one \(M\) for a fixed \(y\), to

\[
\max_{y\in\mathcal{Y}}\ (\alpha\circ y)^\top\big((\lambda+1)K+\mathbf{1}\mathbf{1}^\top\big)(\alpha\circ y).
\tag{23}
\]

That is still combinatorial. The paper replaces the Euclidean norm of \(\sum_i y_i\alpha_i\tilde\phi(x_i)\) by an \(\infty\)-norm. If \(z_{ij}\) is coordinate \(j\) of an empirical feature map of \(\tilde K=(\lambda+1)K+\mathbf{1}\mathbf{1}^\top\), each coordinate produces two candidate labelings: sort the unlabeled points by \(\alpha_i z_{ij}\), and by \(-\alpha_i z_{ij}\), and keep the balance constraint. The candidate with the largest objective is kept, and its matching \(h\) is appended.

The release implements that approximation as follows (`train_shfa_pnorm.m`).

1. Build \(\tilde K\)'s symmetric square root once (`aug_features = real(sqrtm((1+sigma)*K + ones))`, with `sigma = λ = 100`).
2. Initialize \(y^u\) from a \(\rho\)-SVM trained on the labeled target block only, then clip it to the prior ratio. The clip does not refill the ranking: if too many points are positive, the lowest-ranked positives are turned off; if too few are positive, the highest-ranked negatives are turned on.
3. Initialize \(v_1=\sqrt{\lambda/n}\,\mathbf{1}\) and the labeling matrix with that first \(y\).
4. Solve \(p\)-MKL on the paired columns \((v_r, y_r)\). The C routine `sumkernels.cpp` builds

\[
\sum_r d_r\Big(
\big((K^{1/2}v_r)\circ y_r\big)\big((K^{1/2}v_r)\circ y_r\big)^\top
+(K+\mathbf{1}\mathbf{1}^\top)\circ(y_r y_r^\top)
\Big).
\]

5. Score every empirical-map coordinate and its negation, clip each candidate with the same ratio rule, and keep the candidate that maximizes \(\big|\sum_i y_i \alpha_i z_{ij}\big|\).
6. Append that \(y\) and the corresponding \(v=\sqrt{\lambda}\,K^{1/2}(\alpha\circ y)/\|\cdot\|\).
7. Stop on the same relative objective test, or at 50 outer iterations (`hfa_iter = 50`, `hfa_tau = 0.001`).

The bias used at test time is not LIBSVM's `rho`. The code recomputes

\[
\rho=\mathrm{mean}_{i:\alpha_i>0}\big(K_{\mathrm{svm}}\alpha\big)_i
\]

and the demo divides the summed kernel score by this \(\rho\):

```matlab
dec_values = dec_values / rho;
```

Because each binary problem has its own \(\rho\), the division changes which class wins the one-versus-all comparison. The reproduction follows the demo, since that is the program whose printed accuracy is 0.572842.

The test kernel is between the test points and every target training point, labeled and unlabeled. The evaluation is transductive: the images being scored were available, without their individual labels, while \(H\) and \(y^u\) were learned. That is the protocol in Section 4.1 of the paper.

## 5. What the official demo runs

`demo.m` and `demo_shfa.m` do not rerun the paper's 20 random splits, and they do not ship the Reuters or Cross-Lingual Sentiment experiments. They run one fixed split of the Office object set:

| Role | Domain | Raw size | Feature dimension | Samples used |
|---|---|---:|---:|---:|
| Source | Amazon | 2,813 | 800 (SURF bag of words) | 20 per class, 620 |
| Labeled target | DSLR | 498 | 600 | 3 per class, 93 |
| Test, and unlabeled target for SHFA | DSLR | 498 | 600 | the other 405 |

There are 31 categories, not 10. `return_object_target_index_diff_number.m` hard-codes `CLASS_NUM = 31`, and `categories.mat` lists the 31 Office categories from `back_pack` through `trash_can`. Columns are \(\ell_2\)-normalized before the kernel. Parameters in the two demos:

| | HFA (`demo.m`) | SHFA (`demo_shfa.m`) |
|---|---|---|
| \(C_s, C_t\) | 1, 1 | 1, 1 |
| \(C_u\) | — | \(10^{-3}\) |
| \(\lambda\) | 100 | 100 (`sigma`) |
| MKL norm \(p\) | 1 | 1.5 |
| Kernel | Gaussian, mean-distance bandwidth | same |
| Outer iterations / tolerance | 50 / \(10^{-3}\) | 50 / \(10^{-3}\) |

The authors state that this one round prints

- HFA accuracy `0.567901`, which is \(230/405\);
- SHFA accuracy `0.572842`.

Upstream `object_prepare_data.m` used `'../data'`. This tree’s copy under `matlab_implementation/official_shfa_release/` already uses `'./data'`. `demo.m` uses `./data`. The Python port loads `matlab_implementation/official_shfa_release/data` directly.

The solver is cutting-plane MKL around LIBSVM. It is a CPU method. The workstation GPU is irrelevant to the authors' implementation and is not used.

## 6. Complexity

Forming \(K^{1/2}\) on each domain is \(O(n_s^3+n_t^3)\). Each outer iteration trains an MKL model, and each MKL step trains an SVM. Using the usual empirical SVM cost \(O(n^{2.3})\) with \(L\) MKL steps and \(T\) outer iterations, the paper quotes

\[
O(n^3+T L n^{2.3}),
\qquad n=n_s+n_t\ \text{(HFA)},\quad n=n_s+n_t+n_u\ \text{(SHFA)}.
\]

\(T\) and \(L\) stay modest. On this split \(n=713\) for HFA and \(n=1{,}118\) for SHFA, and there are 31 binary models.

## 7. What the paper then compares against

On this supervised protocol the baselines are SVM trained on the labeled target only (SVM\(_T\)), HeMap, DAMA, and ARC-t. SHFA is additionally compared with transductive SVM, which can use the unlabeled target points but cannot use the source, because the source features have a different dimension. The paper reports mean accuracy and standard deviation over 20 splits on Office (Table 4), and over 10 splits on Reuters multilingual text and cross-lingual sentiment (Tables 5 and 6). Those means are not recomputed here: the public code release contains one Office split and neither text collection.

The qualitative claims that do not depend on a particular table entry are:

- Learning \(P\), \(Q\), and the max-margin classifier together beats projecting first and classifying afterwards on these tasks.
- Keeping the original features in the augmented map beats using only the common-subspace block (their SHFA versus SHFA\(_\mathrm{commFeat}\)).
- Unlabeled target samples help most when labeled target data are scarce. On Office, with 3 labeled DSLR images per class, SHFA at \(p=1.5\) improves on HFA by about 1.2 and 1.6 percentage points for the Amazon and Webcam sources in the paper's 20-split average.
- Results are stable for \(\lambda\in[0.01,100]\) and degrade if \(C_u\) is set as large as \(10^{-1}\), because the inferred unlabeled labels are not reliable enough to dominate the loss.
