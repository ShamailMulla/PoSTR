# Approximate Variational Inference & MC-Dropout

Theoretical foundation for treating [[bayesformer|BayesFormer]] dropout as Bayesian inference.

---

## The Goal

We want to know the posterior over network weights given data:
```
p(W | X, Y)  — intractable (high-dimensional, no closed form)
```

Instead, approximate it with a simpler distribution `q(W)` by minimising KL divergence:
```
min_q  KL(q(W) || p(W | X, Y))
```

---

## Variational Inference Derivation

Minimising the KL is equivalent to maximising the Evidence Lower BOund (ELBO):
```
min_q  -Σ ∫ q(W)·log p(y_i | x_i, W) dW  +  KL(q(W) || p(W))
```

The first term is reconstruction loss; the second is a regularisation term.

For the Bernoulli variational distribution used in [[bayesformer|BayesFormer]]:
```
W_{*,i} ~ p·N(0, σ²I) + (1-p)·N(M_{*,i}, σ²I)
```

This reduces to standard **cross-entropy loss + L2 weight regularisation** — exactly what we'd train with anyway. Thus, standard dropout training IS approximate Bayesian inference.

---

## MC-Dropout for Uncertainty Estimation

At test time, run T stochastic forward passes (each with a fresh dropout mask):
```
p(y* | X*) ≈ (1/T) Σ_{t=1}^{T} f_{y, Ŵ_t}(X*)
```

Uncertainty is estimated from the variance across these T passes.

In [[btrl|BTRL]], this is the mechanism for [[thompson-sampling|Thompson Sampling]]:
- Each forward pass with a different locked mask = one sample M̂ from the posterior
- Variance across masks = uncertainty about the world model

---

## BayesFormer's Key Theorem

> Theorem 1: For any input X,
> 1. Computing f_{y,W̃}(X) is equivalent to a forward pass applying dropout with probability p
> 2. The posterior p(y | X*) ≈ (1/T) Σ f_{y,Ŵ_t}(X*), where the predictive uncertainty can be approximated by empirical confidence intervals of the T forward passes

---

## Neural-Linear Approach (PSDRL)

[[psdrl|PSDRL]] uses a different approximation: Bayesian Linear Regression over the last layer only.

Given feature map `φ: X → R^k` from a trained neural network, the output layer weight posterior is:
```
Σ_j^{-1} = σ_j^{-2}·I + σ^{-2}·Φ^T·Φ
μ_j = σ^{-2}·Σ_j·Φ^T·t_j
```

This is closed-form, cheap to compute, but only captures uncertainty in the final linear layer. The hidden representations `φ(x)` are treated as fixed.

**BTRL advantage:** BayesFormer captures uncertainty in **all layers** — Q, K, V matrices, embeddings, FFN. Richer posterior representation but requires [[episode-locked-dropout|episode-locked dropout]] to use correctly.

---

## Dropout Sites and Independence

The key [[bayesformer|BayesFormer]] insight: in standard attention, Q, K, V are identical (`X_attn = X_query = X_key = X_val`), so a single dropout mask applied to the input effectively regularises all three. BayesFormer instead applies **three independent masks** to Q, K, V — treating them as projecting onto different random subspaces:

```
h_query: dropout mask for Q   |  independent
h_key:   dropout mask for K   |  independent  
h_val:   dropout mask for V   |  independent
```

This is interpretable as sampling random low-dimensional projections of the attention space, similar to efficient Transformer architectures like Linformer.

---

## Links

- [[bayesformer]] — applies this theory to Transformers
- [[psdrl]] — uses neural-linear approximation instead
- [[episode-locked-dropout]] — how the posterior samples are used in RL
- [[thompson-sampling]] — why sampling from the posterior enables exploration
