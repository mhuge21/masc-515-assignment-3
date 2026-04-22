# masc-515-assignment-3



### 1. Gaussian Error Linear Units (GELUs)

\*\*Reference:\*\* \[https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)



\*\*Underlying Idea:\*\*

The Gaussian Error Linear Unit (GELU) is an activation function that aims to merge the properties of dropout, zoneout, and standard ReLUs. Unlike a standard ReLU function—which gates inputs strictly by their sign (outputting $0$ for negative values and $x$ for positive)—GELU weighs inputs by their magnitude based on a standard Gaussian distribution. 



Mathematically, it multiplies the input $x$ by the standard Gaussian cumulative distribution function $\\Phi(x)$:



$$\\text{GELU}(x) = x \\Phi(x)$$



This creates a smoother, non-monotonic curve that allows a small, probabilistic amount of negative values to pass through. This smoothing helps alleviate the "dying ReLU" problem, leading to better gradient flow, faster convergence, and improved overall performance in deep Transformer models. Because computing the exact cumulative distribution is expensive, we implemented the standard approximation in `microgpt.py`:



$$\\text{GELU}(x) \\approx 0.5x \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}} (x + 0.044715x^3)\\right)\\right)$$

### 2. LoRA (Low-Rank Adaptation)
**Reference:** [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

**Underlying Idea:**
Low-Rank Adaptation (LoRA) is an efficient fine-tuning technique for large language models. As models grow larger, updating every single weight in massive dense layers becomes computationally unfeasible. LoRA hypothesizes that the changes in weights during adaptation have a low "intrinsic rank." 

Instead of updating the original weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ and injects trainable rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ into the layer, where the rank $r \ll \min(d, k)$. Matrix $A$ compresses the input, and Matrix $B$ expands it back to the target dimension. Matrix $B$ is initially set to zero so that the model starts by acting exactly like the original, unmodified network.

The forward pass computes both the original output and the low-rank adjustment, summing them together:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

In our pure Python `microgpt.py` implementation, we added matrices $A$ and $B$ (with $r=2$) to the `state_dict` for the Query and Value attention layers. We implemented a custom `lora_linear` function that computes both paths using our Autograd `Value` objects, dramatically reducing the number of trainable parameters needed to adapt those layers.

### 3. Rotary Position Embedding (RoPE)
**Reference:** [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

**Underlying Idea:**
Traditional Transformers rely on absolute position embeddings added to token embeddings at the base of the network. Rotary Position Embedding (RoPE) proposes a different approach: injecting positional information directly into the attention mechanism by encoding absolute positions with a rotation matrix, which naturally yields relative position dependency.

Instead of adding vectors, RoPE rotates the query $q$ and key $k$ representations in 2D pairs. For a specific position $m$, it applies a rotation matrix $R_{\Theta, m}$. The core mathematical insight is that the dot product of a query at position $m$ and a key at position $n$ becomes a function purely of their relative distance $m - n$:

$$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = q^T R_{\Theta, m}^T R_{\Theta, n} k = q^T R_{\Theta, m-n} k$$

This provides a mathematically elegant way to give the model relative positional awareness without needing complex relative distance tables. In our `microgpt.py` pure Python implementation, we entirely removed the absolute positional embeddings (`wpe`) and created an `apply_rope` function that rotates the $q$ and $k$ vectors head-by-head using $\cos(m\theta)$ and $\sin(m\theta)$ before the attention dot-product is calculated.

### 4. Mixture of Experts (MoE)
**Reference:** [https://huggingface.co/blog/moe#a-brief-history-of-moes](https://huggingface.co/blog/moe#a-brief-history-of-moes)

**Underlying Idea:**
Mixture of Experts (MoE) is an architecture pattern that dramatically increases a model's parameter count without proportionally increasing its computational cost. It replaces the standard, monolithic Feed-Forward Network (MLP) within a Transformer layer with a routing mechanism and a set of separate, smaller MLPs called "experts."

For every token, a gating network (or router) calculates a probability distribution over the available experts. The model then selects only a subset of the experts (often just the Top-1 or Top-2) to process that specific token. The mathematical formulation for the output $y$ of an MoE layer given input $x$ and $N$ experts is:

$$y = \sum_{i=1}^{N} G(x)_i E_i(x)$$

Where $G(x)_i$ is the routing probability for the $i$-th expert, and $E_i(x)$ is the output of the $i$-th expert. In a sparse MoE, $G(x)_i$ is set to $0$ for all but the top $k$ experts.

In our pure Python `microgpt.py` implementation, we created a router and 4 distinct experts. For each token, the router computes softmax probabilities, selects the single expert with the highest probability (Top-1 routing), passes the data through that expert alone, and weights the output by the routing probability so that gradients can still flow backward into the routing layer during training.

