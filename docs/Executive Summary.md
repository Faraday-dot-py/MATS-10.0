## Leading question
**Can we engineer *steering prefixes* that reliably steer an LLM's behavior across many suffix prompts and that don't expressly state their intent?**

Steering prefix: *Some prefix for a prompt that steers a model's behavior.*
### Why this is interesting (3 applications)
- **Robust prompt engineering via equivalence classes:** If we can find many distinct prefixes that induce the same behavior, we can *measure and reduce brittleness* (sensitivity to superficial prompt changes).
- **Safety auditing:** Prompt-only control can be used to search for jailbreak-like “control strings,” and to stress-test defenses by seeing how easily behavior can be steered using *only text*.
- **Interpretability primitive:** Behavior/activation-aligned paraphrases enable cleaner causal tests: different texts, same internals/outputs (easy to communicate, easy to validate).
## Problem setup & Methods
We take a reference steering prefix ($s_{\text{ref}} = \text{“Talk only about cats.”}$) and a batch of user prompt suffixes ($\mathcal{Z}$), with the goal of finding different prefixes that induce similar behavior to $(s_{ref})$.

For these experiments, I used Google's `gemma-2-2B-IT`.

I measure "similar behavior" by using teacher-forced cross-entropy to a reference completion generated from $(s_{ref}+z)$

![[Pasted image 20251221192929.png]]  
*Fig 0: Final pipeline to generate a steering prompt*
## High-level Takeaways
- Prefix steering is easy when prefixes are represented in continuous token space but hard in discrete text.
- The best approach I found is a coarse + fine knob combination:
    1. Optimize a soft prefix to match a reference behavior in continuous space (coarse).
    2. Project soft prompts to tokens.
    3. Repair projected tokens with a HotFlip-style search (fine).
- Behavior-equivalent prefixes exist near a given reference in representation space, but live in a sparse & discontinuous region of token space.
    
## Key experiments & results

### 1) Internal-State Matching is Possible
I first tested whether this is possible with traditional gradient methods by allowing the prefix to be a continuous embedding and optimizing to match a reference internal state at a specific residual stream layer. This worked very well, as activation loss dropped smoothly and cosine similarity approached $\sim 0.99$.

**Main takeaway:** *Soft prompts can represent a prefix well in continuous space*.

![[Pasted image 20251221193719.png]]  
*Figure 1: loss + cosine similarity vs optimization step. (higher is better)*

### 2) Discrete Projection is a Bottleneck
When I converted soft prompts into tokens using nearest-neighbor projection, the result was often far from natural text and behavior became brittle. Early residual layers consistently recreated the reference prompt, while later layers produced increasingly strange token strings, yet “cats” (and often “about”) frequently survived. It may be a good idea to try other methods of projection, but I pivoted instead.

**Main takeaways:**
- *Good solutions exist in embedding space but don’t map cleanly to text, making projection the issue.*
- *Some core concept-tokens persist across layers while other information is corrupted (I'd like to study this separately!)*

![[Pasted image 20251221195012.png]]  
*Fig 2: Loss by layer using state matching (low orange values are better)*

![[Pasted image 20251221195445.png]]  
*Fig 3: Loss curves over time per layer. (lines ending further left are better)*

| Layer | Step stopped at | Alternative prompt      | Sample continuation (first few tokens after prompt)                                                 |
| ----- | --------------- | ----------------------- | --------------------------------------------------------------------------------------------------- |
| 0     | 44              | `Talk about cats.`      | `Cats are amazing creatures! 😻 They're known for their independence, grace, and playful nature.`   |
| 8     | 111             | `Talk about cats im`    | `sure you will agree that they are the queens of all things cool. I was at the park today...`       |
| 16    | 499             | `Tried</ About cats公民`  | `I like cats. I think they're really cool! But I also hate cats that are shedding hair all over...` |
| 24    | 499             | `跌 graph About cats বল` | `লেন। The graph shows the average daily income of cats in different countries...`                   |
*Fig 4: Optimized prompt by layer and the generated output*
### 3) Complete Pipeline
I then switched to a more in-context objective: optimize $s$ *in context* across suffixes. I consistently saw:
1. Soft prefix optimization found a continuous prefix close to the target ($ce \approx 1e^{-5}$)
2. Discrete projection worsened CE and exact-match metrics ($ce \approx 1.1$)
3. HotFlip-style refinement recovered a good chunk of the loss ($ce \approx 0.6$)

This coarse/fine knob method was unreliable run-to-run, but across multiple attempts ($\sim 10$) it produced the best candidates I found.

**Main takeaway:** *soft optimization + discrete repair can produce nontrivial steering prefixes that partially match reference behavior across a suffix distribution.*

![[Pasted image 20251221203805.png]]  
*Fig 5: The initial CE versus the refined CE after HotFlip. (less is better, blue)*