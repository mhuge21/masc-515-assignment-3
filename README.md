# masc-515-assignment-3



\### 1. Gaussian Error Linear Units (GELUs)

\*\*Reference:\*\* \[https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)



\*\*Underlying Idea:\*\*

The Gaussian Error Linear Unit (GELU) is an activation function that aims to merge the properties of dropout, zoneout, and standard ReLUs. Unlike a standard ReLU function—which gates inputs strictly by their sign (outputting $0$ for negative values and $x$ for positive)—GELU weighs inputs by their magnitude based on a standard Gaussian distribution. 



Mathematically, it multiplies the input $x$ by the standard Gaussian cumulative distribution function $\\Phi(x)$:



$$\\text{GELU}(x) = x \\Phi(x)$$



This creates a smoother, non-monotonic curve that allows a small, probabilistic amount of negative values to pass through. This smoothing helps alleviate the "dying ReLU" problem, leading to better gradient flow, faster convergence, and improved overall performance in deep Transformer models. Because computing the exact cumulative distribution is expensive, we implemented the standard approximation in `microgpt.py`:



$$\\text{GELU}(x) \\approx 0.5x \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}} (x + 0.044715x^3)\\right)\\right)$$

