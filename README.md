# masc-515-assignment-3



\### 1. Gaussian Error Linear Units (GELUs)

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

