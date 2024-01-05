4 Continuous operation
The continuous VO pipeline is the core component of the proposed VO implementation. Its responsibilities are three-fold:
1. Associate keypoints in the current frame to previously triangulated landmarks.
2. Based on this, estimate the current camera pose.
3. Regularly triangulate new landmarks using keypoints not associated to previously triangulated landmarks.
We recommend to implement this in a Markov way using the data flow / function design shown in Figure 2. Formally, we define as $S^i$ the state of the current frame, whose contents are specified further below. Then, we can define a function for processing incoming frames, updating $S^i$ and returning the pose $T_{W C}^i$ as follows:
$$
\left[S^i, T_{W C}^i\right]=\operatorname{processFrame}\left(I^i, I^{i-1}, S^{i-1}\right)
$$

The key idea in this design is that the function inputs solely depend on the output of the previous function call (and the new frame to process), i.e. it has the Markov property. That means we don't need to build a data structure to maintain the history of the past frames, all that is needed is contained in the state $S^i$.
4.1 Associating keypoints to existing landmarks

This can be achieved like in exercise 7. Remember that there, the function ransacLocalization took as input a query image, here $I^i$, a database image, here $I^{i-1}$, keypoints in the database image, 3D landmarks and the projection matrix. To fit this with the Markovian design, we add to $S^i$ the keypoints in the $i$-th frame and the 3D landmarks associated to them. We denote the keypoints with $P^i=\left\{p_k^i\right\}, k \in[1, K], K$ being the keypoint count. The $3 \mathrm{D}$ landmarks are denoted with $X^i=\left\{x(p) \forall p \in P^i\right\}$, meaning that $x\left(p_k^i\right)$ is the 3D landmark associated to $p_k^i$.

We could now propagate $S^i$ according to exercise 7: Obtain $P^i$ from Harris corners and set each landmark $x\left(p_a^i\right)=x\left(p_b^{i-1}\right)$ if the keypoint $p_a^i$ is matched to $p_b^{i-1}$, and if this matching is a RANSAC inlier. However, we strongly recommend to use KLT tracking (see exercise 8): instead of independently extracting $P^i$ and then matching them, keypoints can be tracked from $I^{i-1}$ to $I^i$. This tends to have far fewer outlier associations than extracting keypoints from scratch and then matching them. Furthermore, KLT can track the position of keypoints to sub-pixel accuracy, which leads to more accurate pose estimations. However, KLT is still not perfect, and you might have dynamic objects in the environment that would corrupt your pose estimate, so you still need the RANSAC step.

Note that with that, not all $p_k^i \in P^i$ might end up with an associated landmark. Those that do not can be discarded: they will no longer be useful for $I^{i+1}, I^{i+2} \ldots$ As a consequence, $P^i$ will shrink over time, unless we actively expand it, which is discussed in Section 4.3. For KLT we recommend to use the Matlab class vision.PointTracker .

4.2 Estimating the current pose

Whether you obtain keypoint-to-landmark associations from patch matching or from KLT tracking, you should jointly estimate pose and inliers using RANSAC, like in exercise 7. This will give you $T_{W C}^i$ as an automatic by-product. In exercise 7, we had suggested to refine the P3P guess with a DLT solution for all inliers, once the maximum set of inliers has been determined. However, while developing the reference VO, we have found that the DLT solution is very often worse than the best P3P guess.

To sum up, so far $S^i=\left(P^i, X^i\right)$. We recommend to use for $S^i$ a Matlab struct and a $2 \times K$ and $3 \times K$ matrix for $P^i$ and $X^i$, respectively.
4.3 $\quad$ Triangulating new landmarks

So far, the pipeline can use the landmarks $X^1$ from the initialization phase to localize subsequent frames. However, once the camera has moved far enough, these landmarks might not be visible any more. It is thus necessary to continuously create new landmarks. We propose an approach which maintains the Markov property of our design and provides new landmarks asynchronously, as soon as they can be triangulated reliably.

The idea is to initialize, for each new frame $I$, a set of candidate keypoints, and try to track them through the next frames. Thus, at every point in time, we maintain a set of candidate keypoints $C^i=\left\{c_m^i\right\}, m \in[1, M]$ which have been tracked from previous frames. Let us define $\Psi_i^{i+1}\left(c^i\right)$ as the expression for the keypoint that is obtained from tracking $c^i$ from frame $I^i$ to $I^{i+1}$. Then, we assume that this operation can be inverted such that
$$
\Psi_{i+1}^i\left(\Psi_i^{i+1}\left(c^i\right)\right)=c^i
$$
and define the concatenation of tracking a keypoint across several frames as
$$
\Psi_i^{i+n}\left(c^i\right)=\Psi_{i+n-1}^{i+n}\left(\Psi_{i+n-2}^{i+n-1}\left(\ldots \Psi_i^{i+1}\left(c^i\right)\right)\right),
$$
which can also be inverted in the sense of (2). For every candidate keypoint $c^i \in C^i$, we call the sequence $\Gamma\left(c_m^i\right)=\left\{\Psi_i^{i-L_m}\left(c_m^i\right), \Psi_i^{i-L_m+1}\left(c_m^i\right), \ldots, c_m^i\right\}$ of tracked keypoints from frame $I^{i-L_m}$ to frame $I^i$ a keypoint track of length $L_m$. As soon as a given keypoint track $\Gamma_m$ meets some conditions (more details on that below), we can reliably triangulate a new landmark from the keypoint observations $\left\{\Psi_i^{i-L_m}\left(c_m^i\right), \Psi_i^{i-L_m+1}\left(c_m^i\right), \ldots, c_m^i\right\}$, and the corresponding camera poses $\left\{T_{W C}^{i-L_m}, T_{W C}^{i-L_m+1}, \ldots, T_{W C}^i\right\}$. To simplify, we assume that a good enough triangulation for a given track can be achieved using only the most recent observation $c_m^i$, the first ever observation of the keypoint $f\left(c_m^i\right):=\Psi_i^{i-L_m}\left(c_m^i\right)$, and the corresponding poses $T_{W C}^i$ and $\tau\left(c_m^i\right):=T_{W C}^{i-L_m}$. Hence, all we need to remember of the track for a given keypoint $c_m^i$ is $f\left(c_m^i\right)$ and $\tau\left(c_m^i\right)$.
We can add the following data to the state $S^i$ to reflect this:
- The set of candidate keypoints $C^i$.
- A set containing the first observations of the track of each keypoint $F^i:=\left\{f(c) \forall c \in C^i\right)$.
- The camera poses at the first observation of the keypoint $\mathcal{T}^i:=\left\{\tau(c) \forall c \in C^i\right\}$.

With this, the state ends up being:
$$
S^i=\left(P^i, X^i, C^i, F^i, \mathcal{T}^i\right)
$$
$C^i, F^i, \mathcal{T}^i$ can respectively be represented as matrices of shape $2 \times M, 2 \times M, 12 \times M$ (or $16 \times M$ for the latter), where the transformation matrices $\tau \in \mathcal{T}^i$ are reshaped to vectors.

To propagate the new components of this state, we can track $C^i=\left\{\Psi_{i-1}^i(c) \forall c \in C^{i-1}\right\}$ in the same way that $\left\{p_k^i\right\}$ are tracked in Section 4.1 (we had not used this formalism there yet). Trivially, $F^i$ is defined by $f(c)=f\left(\Psi_i^{i-1}(c)\right) \forall c \in C^i$, and $\mathcal{T}^i$ analogously. If a candidate keypoint $c \in C^{i-1}$ fails to be tracked, it and $(f(c), \tau(c))$ get discarded. As a consequence, $C^i, F^i, \mathcal{T}^i$ also shrink over time. To mitigate this, you will need to continuously add newly detected keypoints $c^{\prime}$ to $C^i$ and set

the corresponding $\left(f\left(c^{\prime}\right), \tau\left(c^{\prime}\right)\right)=\left(c^{\prime}, T_{W C}^i\right)$. Consider making sure that these newly added keypoints are not redundant with existing keypoints $C^i$ and $P^i$.

Finally, to obtain new 3D landmarks, you can attempt to triangulate a landmark $x$ from each $(c, f(c), \tau(c)) \forall c \in C^i$ similarly as the triangulation in exercise 5. However, you should require a minimum baseline to make sure that the triangulations you get have good quality. We recommend finding a threshold for the angle $\alpha(c)$ between the bearing vectors corresponding to the keypoint observations and camera poses, see Figure 3. If the angle exceeds that threshold, you can remove $(c, f(c), \tau(c))$ from $C^i, F^i, \mathcal{T}^i$ and append $(c, x(c))$ to $P^i, X^i$.
5 General hints
- In general, proceed step by step and verify your intermediate results visually. For example, make sure that matching, localization and landmark propagation work properly before triangulating new landmarks.
- You can save a lot of tedious coding and avoid bugs if you learn to master Matlab indexing. We recommend that you carefully read and try out all of that tutorial except the advanced examples using linear indexing. Don't forget the logical indexing section.