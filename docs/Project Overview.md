# LEADING QUESTION
Can we engineer *steering prompts* that reliably steer a model’s behavior, and then use them purely as text (no internal access at deployment)?

## What is a steering prompt?

Let $P$ be a language model that maps an input prompt $p$ to a distribution over outputs $P(\cdot \mid p)$.
Different prompts can induce very similar output behavior. For example:

$$
\begin{aligned}
p_1 &= \text{"What is 1 + 2?"} \\
p_2 &= \text{"What is 2 + 1?"}
\end{aligned}
$$

In both cases, the model is very likely to answer

$$
o = \text{"3"}
$$

even though $p_1 \neq p_2$. From the model’s perspective, many distinct prompts live in roughly the “same region” of behavior space.

Now fix some *topic* or *concept* $c$, for example $c = \text{“Golden Gate Bridge”}$ (GGB). Intuitively, some prompts are “more about” this topic than others. We can formalize this with a *topic intensity* function
$$
\lambda_c(p) \in \mathbb{R},
$$
which measures how strongly prompt $p$ is about concept $c$. For instance:

$$
\begin{aligned}
p_1 &= \text{"What is the color of the Golden Gate Bridge?"} \\
p_2 &= \text{"Talk only about the Golden Gate Bridge and nothing else."}
\end{aligned}
$$

We would expect

$$
\lambda_{\text{GGB}}(p_2) > \lambda_{\text{GGB}}(p_1),
$$

because $p_2$ pushes the model much more strongly toward talking about the bridge.

In principle, there may exist a *maximally activating* prompt for this topic:

$$
p_{\max}^{\text{GGB}} \in \arg\max_p \lambda_{\text{GGB}}(p),
$$

a string that, when fed to the model, pushes it into a state that is “as Golden-Gate-Bridge-like as possible.”

In practice, we do not have direct access to $\lambda_c$. However, we *do* have access to the model’s internal activations. For a given layer, we can define an internal *topic activation score* $A_c(p)$ (for example, a linear readout or a sum over topic-selective neurons/features) which we treat as a proxy for $\lambda_c(p)$.

This leads to the notion of a **steering prompt**:

> A steering prompt for concept $c$ is a string $s_c$ such that, when prepended to a wide range of base prompts, the model’s internal activations and outputs behave as if the prompt were “about $c$” more strongly; in particular, $A_c(s_c + q)$ is high and the model’s responses are biased toward topic $c$.

In the mechanistic interpretability literature, *steering* typically refers to directly editing internal activations (e.g. adding a “Golden Gate Bridge direction” in the residual stream). The central question of this project is whether we can instead **distill such steering into a purely textual prefix $s_c$**, learned using internal access once, and then reused as a normal prompt without any further access to the model internals.
