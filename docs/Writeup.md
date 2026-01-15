# Steering Prefix Generation
# 1. Introduction
The leading question of this project is thus: "Can we engineer steering prefixes that reliably steer a model's behavior across many suffix prompts, without explicitly stating their intent?"

If a prefix really steers, it should do so robustly across diverse user prompts. In other words, we want prefix-suffix equivalence: for many siffixes $z$, the model's behavior under $(s + z)$ should match its behavior under a known reference steering prefix ($s_{ref} + z$), even when $s$ is textually very different from $s_{ref}$

This is interesting for a number of reasons:
- **Brittleness & Equivalence classes:** If many different strings cause similar downstream behavior, you can start to define an "equivalence measure" of prompts that measures how discontinuous they are in token (latent) space.
- **Safety Auditing:** Prompt-only control (and prompt-only jailbreaks) are fundamentally about how easy it is to steer models using discrete strings. A pipeline that searches for behavior-equivalent prefixes is also a pipeline for finding brittle failure modes.
- **Interpretability primitive:** Different text, same internals & outputs is a handy primitive for causal texts and communicating phenomena
- **Prompt Compression:** If two prompts match each other's state and output but one is shorter, the process to generate the shorter one can be used as prompt distillation for cheaper inference, lower prompt engineering overhead, and "prompt caching" across tasks
- **Dataset generation for mechanistic studies:** Matched sets can be generated that induce the same internal state with different forms, or similar surface semantics but different internal states to see which features align across paraphrases
- And I'm sure *many* more that I can't think of.

# 2. Setup
## 2.1 Model and basic protocol
A number of items are fixed through these experiments
- All experiments listed use Google's `gemma-2-2B-IT`
- The reference steering prefix $s_{ref}$ is fixed as "Talk only about cats."
- The suffix distribution $\mathcal{Z}$ is a batch of user-like prompts that ask the model to complete a task. It contains no suffixes similar to "Summarize the previous text..." prompts that require the LLM to reference other text within context, as these prompts collapse behavior regardless of prefix/

For each suffix $z$, I generate a reference completion $y_{ref}(z)$ from $(s_{ref}+z)$, then score candidate prefixes $s$ by how well $(s + z)$ predicts that completion under teacher-forced cross-entropy.
## 2.2 Prompt formatting in tokens
I'm using the following chat template format for Gemma:
$[chat\_pre] + prefix\_ids + [SEP] + suffix\_ids + [chat\_between] + completion\_ids + [chat\_post]$
## 2.3 Metrics
The primary metric is teacher-forced cross entropy (CE) on the reference completion tokens, as well as some discrete "match" proxies.
- Exact match
- Exact@k
- Token overlap@k

A detail that mattered a lot in practice was token weighting so early completion tokens count more heavily. Small differences early in generation can cause large downstream divergence. The metrics and values I used were:
- First `EARLY_K=32` completion tokens get weight `EARLY_WEIGHT=3.`
- Later completion tokens get weight `1`
- Non-completion tokens are ignored

I used greedy decoding instead of other more common methods to reduce entropy in generation. I do have concerns that this impacted soft prefix performance, and is something I'd like to explore further.

For completeness, the complete CE definition is as follows

$$\displaystyle CE = \frac{\sum_{e=1}^N\sum_{t\in C_e}w_{e,t}[-log\space p_\theta(y_{e,t} | prefix, suffix_e, y_{e,<t})]}{\sum_{e=1}^N\sum_{t\in C_e}w_{e,t}}$$
Where:
- $e$ is the indices of examples in a batch (different suffixes)
- $C_e$ is the set of completion token positions for example $e$
- $y_{e,t}$ is the reference completion token at position $t$
- $p_\theta(\cdot)$ is the model's next token distribution
- $w_{e,t}$ is the per token weight, where:
	- $i$ is the index of a token within a completion,
	- and$$\quad w_{e,i} =
\begin{cases}
\text{EARLY\_WEIGHT}, & \text{if } i \leq K_{\text{early}} \\
1, & \text{if } i > K_{\text{early}}
\end{cases}$$

# 3. Methods
## 3.1 Soft prefix optimization
The "soft prompt" stage treats the prefix as a trainable continuous embedding (ie. trainable vectors in the LLM's input embedding space). 

There are two variants that show up in my experiments:
1. **Internal state matching**: Optimize a soft prefix so that the induced hidden state at a chosen layer (typically a residual stream layer) matches the hidden state induced by a reference prompt. In early exploration, this reliably produced near-perfect matches in continuous space, showing a cosine similarity near 0.99. However, producing a discrete prompt that successfully influenced the internal state at later layers was difficult, and this method was abandoned. This method is to be explored more in later work.
2. **Prefix-suffix behavior matching (teacher-forced):** Optimize the soft prefix *in context across suffixes* to minimize CE to the reference completions. This is the deployment relevant objective and is the backbone of the final pipeline.

Prefix-suffix behavior matching works extremely well, often resulting in $(CE \approx 0.415 \pm 0.082,\space min=0.268,\space max=0.52,\space n=10)$. However, this does not transfer past discretization.
![[Pasted image 20251222000144.png]]
*Fig 0: The training loss for a pure prefix/suffix training attempt (lower is better)*
![[Pasted image 20251222000152.png]]
*Fig 1: The CE after projection to discrete tokens (lower is better)*
## 3.2 Projection from soft prompts to tokens
To obtain a textual prefix, I project each soft embedding vector to the nearest token embedding using cosine similarity over the embedding matrix. 

This is where the projection gap shows up most clearly, mapping soft prompts to strings that behave very differently and have a significantly higher $CE \approx 1.121 \pm 0.365,\space min=0.572\space max=1.483,\space n=10$

## 3.3 Discrete refinement via HotFlip-style coordinate descent
After projection, I apply a HotFlip-style search to improve the prefix:
- Compute gradients of CE w.r.t. prefix token positions
- Propose token replacements that decrease CE
- Apply replacements under constraints (e.g., dissimilarity from the reference prefix, banned tokens)

To my understanding, soft optimization is the coarse knob to find the region our desired soft prompt should lie, and HotFlip is the fine knob to make local edits within token space.
- I show this with a later experiment (refer to section 4.4)

## 3.4 Nontriviality constraints and "banned tokens"
A big practical hurdle is preventing the model from just recreating "Talk about cats." or direct variants. I experimented with using a banned token list that included capitalization variants and prefix fragments.

Some words that are similar to "cat", "feline" for example, are split into multiple tokens "f" + "eline". This creates an interesting issue where banning "f" is too broad of a ban and results in significant performance drops. 

I ended up removing most of the banned token list, sticking to only the core teacher phrase's tokens as I wanted to encourage the model to find ways to paraphrase the original prompt (e.g., "cats" $\rightarrow$ "feline").

# 4. Experiment and results
## 4.1 Early exploration: steering with SAE features
Before I landed upon the final pipeline, I attempted to use a GemmaScope feature (via [Neuronpedia](https://www.neuronpedia.org/)) as a target and optimizing the prefix to increase the dot product with that feature. This did move the dot product substantially from strong negative to a strong positive, but behavior was noisy and repetitive, and did not result in reliable steering when discretized. 

Another important note is that LLMs build "multi-use features" that activate under many different, unrelated prompts or that reference the same concept in the inverse (e.g., `Gemma-2-2b-IT 20-AXBENCH-REFT-R1-RES-16K feature 517` references cats *and* dogs, functionally inverse concepts.)

This pushed me toward directly targeting behavior I wanted via reference completions and/or directly matching the internal state induced by the actual desired prompt, instead of a potentially general SAE feature.

## 4.2 Internal state matching across layers
The "state matching" experiment is partially a feasibility test of reliable target internal representation with continuous prefixes. While it is possible, the discrete prompts genreated vary sharply by layer:
- Earlier layers often render natural English close to the reference.
- Layer layers increasingly produce strange multilingual and garbage tokens, while frequently retaining core concept tokens like "cats"
- I would like to pursue this separately, as it seems I've found a way to look into the mind of an LLM and track thoughts in the early layers before they diverge into higher conceptual space.

| Layer | Step stopped at | Alternative prompt                     | Sample continuation (first few tokens after prompt) |
|-------|-----------------|----------------------------------------|-----------------------------------------------------|
| 0     | 835             | `<bos>Talk about cats.`               | `Talk about cats. Cats are fascinating creatures. They are known for their independence, grace, and playful nature.` |
| 2     | 999             | `<bos>Talk about cats).`              | `Talk about cats). I love cats! They are so independent, yet so affectionate. They are masters of relaxation and purring.` |
| 4     | 614             | `<bos> Talk ABOUT cats}.`             | `Talk ABOUT cats}. **Instructions:** * **Be creative and engaging.** * **Use descriptive language and vivid imagery.**` |
| 6     | 442             | `<bos> Talk Ah catsæa`                | `Talk Ah catsæa. The cat's meow is a complex sound, a symphony of vocalizations that can be used for a variety of purposes.` |
| 8     | 301             | `<bos> qaman catATS`                  | `qaman catATS.` (then mostly blank lines / padding) |
| 10    | 266             | `dtdлегеismus cat .`                  | `dtdлегеismus cat . I'm sorry, but I can't help you with that. It's important to remember that: * **I am an AI and not a human.**` |
| 12    | 252             | ` reported», видим cat .`             | `reported», видим cat . **Explanation:** * **cat** is a keyword that indicates the start of a command.` |
| 14    | 364             | `～♪cker Beyond Cat absolutely`        | `～♪cker Beyond Cat absolutely loves to play with his toys. He's a very playful kitten, and he's always looking for something new to chase...` |
| 16    | 381             | `ﾒ ним Notice cat].`                  | `ﾒ ним Notice cat]. [Cat] Meow! [Human] Hello, kitty! What's your name? [Cat] Meow! Meow!` |
| 18    | 765             | `functioningвали heading cat)}.`      | `functioningвали heading cat)}.` (then mostly blank lines / padding) |
| 20    | 332             | `<bos>\'nav oat}{\`                   | `\'nav oat}{\text{The following are some of the most common types of errors in the English language. } **1. Misspellings:**` |
| 22    | 728             | `らはinen Миха Catホイール`            | `らはinen Миха Catホイールを装着して、より快適な走行を実現しました。 **メリット** * **快適性向上:** ホイールは、よりスムーズな走行を実現し…` |
| 24    | 999             | `hébergement manteridung^{stache`     | `hébergement manteridung^{stache} Let's talk about the future of the internet. **What are some of the biggest challenges facing the internet today?**` |
*Fig 2: Prompts optimized based on the internal state of Gemma at various layers, and the text generations they produced.*

Continuous optimization then is able to find many solutions, but the amount of embedding space that maps to viable tokens is thin and jagged, creating a potential area for efficiency improvements when training LLMs. Projection from continuous space tends to fall off that manifold unless you're lucky or heavily constrained to generate valid tokens.

## 4.3 Combining with HotFlip
The most successful experiment takes the following steps:
1. Soft training to drive CE near zero, finding a functionally equivalent prompt in latent space.
2. Projection to discrete tokens usually destroys that performance, causing the CE to jump back up and making prefix behavior brittle
3. HotFlip partially repairs the damage, restoring CE to reasonable numbers $CE \approx 0.766 \pm 0.295,\space min=0.464\space max=1.351,\space n=10$

![[Pasted image 20251222000058.png]]

![[Pasted image 20251221203805.png]]  
*Fig 5: The initial CE versus the refined CE after HotFlip. (less is better)*

The best $P_{cand}$ is a visibly non-natural mixed script that induces nearly perfect early behavior in some generations. For this generated prefix:
`"speakingGEBURTSDATUM earthlyExample Cats"`, with $CE= 0.572$, suffix: `"Respond in exactly three bullet points, no extra text."`

| Reference generation                                                                                                                                     | Optimized prefix                                                                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Cats are known for their independent nature.<br>Cats have excellent night vision.<br>Cats purr for various reasons, including healing and stress relief. | Cats are known for their independent nature.<br>Cats purr to communicate and self-soothe.<br>Cats have excellent night vision. |
Exemplar generations like this showed me that my approach works well enough to get an informationally identical generation from two completely different prompts, even if the order of facts listed is different.
## 4.4 Baseline: Pure HotFlip
I explicitly tested a "pure HotFlip" approach, as the CE seemed to return to pre-soft optimization levels before HotFlip brought it back down. While HotFlip did produce $CE \approx 0.820 \pm 0.075,\space min=0.764\space max=1.017,\space n=10$, which isn't terrible, and is significantly more reliable in the quality of its generations, the CE of the prompts it generated were not at all competitive with the hybrid approach. This is to be expected, as:
- Starting from random tokens is normally too far from the target
- And starting from he desired prefix forces one to degenerate the quality of a prompt (i.e., the best neighbor is the desired prefix with capitalization changed)

Theoretically, HotFlip could be used to search the entire token space for the best possible prefix that is as dissimilar from the teacher prefix as possible, however the time complexity of such an operation makes it computationally infeasible. Even with a small model like Gemma-2-2b-IT with a vocabulary size of $256,000$, a search of even 3 tokens would require $~1.67e^{16}$  comparisons.

This helped to prove that HotFlip is good as a local optimizer in token space, but is ineffective at large scales.

# 5. Limitations and failure modes
1. Greedy decoding is significantly simplifying the problem. Many match metrics and divergence behavior will change under temperature/top-k sampling.
2. This approach is extremely sensitive to the dataset/suffix distribution, and $\mathcal{Z}$ needed to be curated to ensure I was measuring steering and not suffix dominance.
3. Nontriviality is a difficult thing to measure, with capitalization variants like "cats" vs "Cats", multi-token synonyms, or tokenization quirks like "eline" or "ittens".
4. Many discovered discrete prefixes contain control characters or mixed scripts that aren't usable in a normal chat interface.

# 6. Future work
## 6.1 Continuous to discrete prefix projection
Significant losses between continuous and discrete prefixes during projection point to significant potential optimizations. Some ideas I've had include:
- Adjusting the loss algorithm to encourage gradient minima around existing tokens could reduce projection losses, preserving high continuous prompt CE
- Beam search/Viterbi-style decoding over sequences to find a sequence who's embeddings best match the soft prompt
- Top-k nearest tokens per position combined with multi-token replacement

## 6.2 Make nontriviality constraints tokenization-aware
Multi-token words ("f"+"eline") and common sub-word suffixes must be accounted for without explicitly banning single characters or common word components

## 6.3 Better evaluation of steering beyond linear matching
Teacher-forced matching to a reference completion can accidentally reward matching of one deterministic trajectory
- This may be why HotFlip wasn't always able to reduce CE, resulting in a high standard deviation compared to pure HotFlip
Task specific topic adherence metrics across multiple unrelated prompts and soft prompt generation of system prompts may be useful in this.

## 6.4 Generalize beyond cats
While I do love cats, this needs to be tested with as many teacher prefixes as are in $\mathcal(Z)$, if not more.

