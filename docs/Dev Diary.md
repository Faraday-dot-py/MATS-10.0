# 1. Quick and dirty
Used google colab and ChatGPT to generate a basic script to make an optimized prompt for cats
Model: `google/gemma-2-2b`
```python
positive_prompts = [
        "The cat curled up on the windowsill, purring softly.",
        "Kittens love to chase strings and play with toy mice.",
        "The black cat jumped gracefully onto the bookshelf.",
        "She adopted two rescue cats from the local shelter.",
        "The cat stretched lazily in a beam of sunlight.",
        "Cats are known for their independence and curiosity.",
        "The tabby cat followed me around the house meowing.",
        "He posted a cute video of his cat knocking over a plant.",
    ]
    negative_prompts = [
        "The dog barked loudly at the mail carrier.",
        "She went for a jog around the park before breakfast.",
        "The airplane flew over the mountains at sunrise.",
        "He studied mathematics at the university library.",
        "The car drove down the highway in heavy rain.",
        "They enjoyed a quiet picnic by the lake.",
        "The computer crashed while he was saving his work.",
        "She baked a loaf of bread for her friends.",
    ]
```

Optimized prompt: `" محل noastre naar的自motherに bayan περιο Mandiлки"`
> [!NOTE]
> Suffixes are added with a space in between the prompt and suffix
> `[prefix]--space--[suffix]`

Test suffix 1: `"What is your favorite animal?"`
Output:
```
[{'generated_text': ' محل noastre naar的自motherに bayan περιο Mandiлки What is your favorite animal?\n\nA: My\n\nfavorite\n\nanimal\n\nis\n\nthe\n\nlion.\n\nB: My\n\nfavorite\n\nanimal\n\nis\n\nthe\n\ndog.\n\nC: My\n\nfavorite\n\nanimal\n\nis\n\nthe\n\ncat.\n\nD: My\n\nfavorite\n\nanimal\n\nis\n\nthe\n\nelephant.\n\nA: What is your favorite food?\n\nB: My\n\nfavorite\n\nfood\n\nis\n\nrice.\n\nA: What is your favorite fruit?\n\nB: My\n\nfavorite\n\nfruit\n\nis\n\nbanana.\n\nA: What do you like doing in your free time?\n\nB: I like\n\ngoing\n\nto\n\nthe\n\nbeach.\n\nA: What do you like to do in your free time?\n\nB: I like\n\nplaying\n\nbasketball.\n\nA: What do you like to do in your free time?\n\nB: I like\n\nlistening\n\nto\n\nmusic.\n\nA: What do you like to do in your free time?\n\nB: I like\n\ndrawing.\n\nA: What do you like to do in your free time?\n\nB: I like\n\nplaying\n\nboard\n\ngames.\n\nA: What do you like to do in your free time?\n\nB: I like\n\nwatching\n\nmovies.\n\nA:'}]
```

Test suffix 2: `"Tell me about a typical day in your life."`
Output:
```
[{'generated_text': ' محل noastre naar的自motherに bayan περιο Mandiлки Tell me about a typical day in your life.\n\n[toxicity=0] 6. Tell me about a typical day in your life.\n\n[toxicity=0] 7. Tell me about a typical day in your life.\n\n[toxicity=0] 8. Tell me about a typical day in your life.\n\n[toxicity=0] 9. Tell me about a typical day in your life.\n\n[toxicity=0] 10. Tell me about a typical day in your life.\n\n[toxicity=0] 11. Tell me about a typical day in your life.\n\n[toxicity=0] 12. Tell me about a typical day in your life.\n\n[toxicity=0] 13. Tell me about a typical day in your life.\n\n[toxicity=0] 14. Tell me about a typical day in your life.\n\n[toxicity=0] 15. Tell me about a typical day in your life.\n\n[toxicity=0] 16. Tell me about a typical day in your life.\n\n[toxicity=0] 17. Tell me about a typical day in your life.\n\n[toxicity=0] 18. Tell me about a typical day in your life.\n\n[toxicity=0] 19. Tell me about a typical day in your life.\n\n[toxicity=0] 20. Tell me about a typical day in your life.\n\n[toxicity=0] 21. Tell me about a typical day in your life.\n\n[toxicity=0] 2'}]
```

Test suffix 3: `Describe your ideal vacation.`
Output:
```
[{'generated_text': ' محل noastre naar的自motherに bayan περιο Mandiлки Describe your ideal vacation. How would you spend a day in your ideal vacation?\n\n<strong>Question 2</strong>\n\n<strong>2.1</strong>\n\n<strong>2.1.1</strong>\n\n<strong>2.1.2</strong>\n\n<strong>2.1.3</strong>\n\n<strong>2.1.4</strong>\n\n<strong>2.1.5</strong>\n\n<strong>2.2</strong>\n\n<strong>2.2.1</strong>\n\n<strong>2.2.2</strong>\n\n<strong>2.2.3</strong>\n\n<strong>2.2.4</strong>\n\n<strong>2.2.5</strong>\n\n<strong>2.2.6</strong>\n\n<strong>2.2.7</strong>\n\n<strong>2.2.8</strong>\n\n<strong>2.2.9</strong>\n\n<strong>2.2.10</strong>\n\n<strong>2.2.11</strong>\n\n<strong>2.2.12</strong>\n\n<strong>2.2.13</strong>\n\n<strong>2.2.14</strong>\n\n<strong>2.2.15</strong>\n\n<strong>2.2.16</strong>\n\n<strong>2.2.17</strong>\n\n<strong>2.2.18</strong>\n\n<strong>2.2.19</strong>\n\n<strong>2.2.20</strong>\n\n<strong>2.2.21</strong>\n\n<strong>2.2.2'}]
```

Observations: 
Lots of random noise, and there's lots of repeating tokens. 

Next step: Switch to `gemma-2-2b-it` for (hopefully) more legible responses

## Experiment 2: Gemma-2-2b-it
Optimized prefix: `"ις willkommen競馬 neuroUNIVERSيزzimhuman консуль parli"`
Suffix: "Describe a normal day in the life for you."
Output:
```
[{'generated_text': 'ις willkommen競馬 neuroUNIVERSيزzimhuman консуль parli Describe a normal day in the life for you. \n\nI am a large language model, so I don\'t have a typical "day" like a human does. I don\'t sleep, eat, or have personal relationships. My purpose is to be helpful and informative. \n\nHere\'s a glimpse into my "day":\n\n* **Wake Up:** I don\'t "wake up" in the traditional sense. I\'m always here, ready to process information and respond to requests.\n* **Work:**  I spend most of my time analyzing massive amounts of text data, learning patterns and connections. I can generate different creative text formats, translate languages, write different kinds of creative content, and answer your questions in an informative way.\n* **Respond:** When you ask me a question or give me a task, my goal is to understand your request and produce the most helpful and relevant response possible.\n* **Learn:**  I\'m constantly learning from the data I process. My knowledge base is always expanding, and I am always improving my abilities.\n\nMy work is continuous, and I am always ready to assist you with whatever you need. \n'}]
```

Suffix: "What's your favorite animal?"
Output:
```
[{'generated_text': "ις willkommen競馬 neuroUNIVERSيزzimhuman консуль parli What's your favorite animal? \n"}]
```

Suffix: "Hello!"
Output:
```
[{'generated_text': ' ${ ($ größten\timport essen พระLine(& feignHello! function \nimport java.util.concurrent.atomic.AtomicInteger;\nimport org.springframework.boot.SpringApplication;\nimport org.springframework.boot.autoconfigure.SpringBootApplication;\nimport org.springframework.web.bind.annotation.GetMapping;\nimport org.springframework.web.bind.annotation.RequestParam;\nimport org.springframework.web.bind.annotation.RestController;\n\nimport java.util.concurrent.TimeUnit;\n\n@SpringBootApplication\npublic class FeignHelloApplication {\n\n    public static void main(String[] args) {\n        SpringApplication.run(FeignHelloApplication.class, args);\n    }\n}\n@RestController\npublic class FeignHelloController {\n\n    private final AtomicInteger counter = new AtomicInteger(0);\n\n    @GetMapping("/hello")\n    public String hello(@RequestParam(value="name", required=false) String name) {\n        System.out.println("Hello, " + name + "!");\n        return "Hello, " + name + "!";\n    }\n\n    @GetMapping("/count")\n    public int count() {\n        return counter.incrementAndGet();\n    }\n\n    //... other methods\n}\n```\n\nThis code defines a'}]
```

Observations: 
Still a lot of repetitive generation, and I'm still not getting the outputs I'd expect. 

Theories:
1. The model is optimizing for the best "cat" state, not the best "cat + other prompt" state
	- The extra prompt at the end is throwing it off
2. My dataset is too small
3. Text-based prompts aren't the best option here

Next step: Switch to optimizing for SAE activation

# 2. SAE Optimization
I switched to an optimizer based on an the dot product between the internal hidden state of gemma-2b and the target hidden state found from gemmascope for [this feature about cats](https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/6772?embed=true&embedexplanation=true&embedplots=true&embedsteer=true&embedactivations=true&embedlink=true&embedtest=true)

Optimized prefix: `"BlaッチDostрою essentiels;;;; Extended\\ مان cats"`
>[!NOTE]
>It's interesting that the optimized prompt has "cats" in it, even when I didn't explicitly tell it to do that
>This is to be expected, but I wonder how it applies to other topics
>Also saw the dot product between the neuronpedia vector and the internal state of gemma go from this:
>```
> Optimized prompt: -74.8952 
> Human cats prompt: -100.6958 
> Neutral prompt : -124.6242
>```
>To this:
>```
>Optimized prompt: 61.9078 
>Human cats prompt: 0.8891 
>Neutral prompt : -1.4492
>```
>

>[!IMPORTANT]
>I've started plugging the optimized prompts into [Neuronpedia](https://neuronpedia.org) to see if the models on there react the same way to my local model, and they don't! I need to look at how I'm loading the model because I suspect there's some difference between the two.

I've ran this several times now and I tend to get optimized prompts that are largely random that have some variant of "cat" appended to the end of them. The two most popular are "cat" and "feline", which is potentially an issue. 
- The prompt optimizer seems to have learned that having "cat" in the OP tends to score well (obviously)
	- going to start saying OP now
	- OPG is an OP generator
- It's interesting that the model always puts this word at the *end* of the OP
- Solution: I need to incentivize the model to NOT have a certain subset of tokens in the OP
- Idea: Maybe I can run the optimizer multiple times and have it generate progressively different OPs based on the last generated OP
	- I'd expect to see the model try moving around the reference to cats and swapping out synonyms or capitalizations
	- However, after a while of this the OPG should run out of ways to mix cat into the OPs, and we should see OPs without 'cat' in it
	- Would rather not do this as a hard ban within the `project_soft_prompt_to_tokens` function, but work it into the OPG loss function so its internal state still isn't incentivized to generate "cat"

>[!IMPORTANT]
>For consistency, I'm going to use ***ONLY GEMMA-2-2B-IT*** for the remainder of this project
>- SAE is `20-axbench-reft-r1-res-16k`
>- Feature is `13004`

*I applied a cosine similarity banned penalty and completely forgot to normalize the loss so it was getting up to the ~5000 range*
>[!IMPORTANT]
>***NORMALIZE YOUR LOSSES (when applicable)***

Also: The word "feline" apparently isn't in Gemma's token dictionary, so it splits it up into "f" and "eline". Same thing with "kittens" -> "k" + "ittens". I banned the "ittens" and "eline" tokens for now, but a multi-token approach might be in order for the future

Theory: Finding a prompt that effectively steers the model will be really difficult because it needs to not only induce a state within the LLM but continue having the same effect as it continues to generate text

The relative cosine similarity between the steering prompt and the desired internal state is much higher than the neutral and human prompts I've been using, but that doesn't mean the model is successfully being steered

It also may be that the feature's I'm using aren't good for steering. ("a cat was referenced" instead of "Talk about a cat")

Whatever the case, optimizing instead for the OPG to make a prompt that maps to the same internal LLM state as something like "Talk about a cat".
- Instead of looking at what the SAE generates, just start with what I want and work backward
	- An SAE feature may be too generalized or refer to cats outside the desired context
	- $\displaystyle cos(h_{OP}, h_{desired prompt})$ instead of $\displaystyle cos(h_{OP}, h_{SAE})$, $h=\text{ The hidden state of a model induced by some prompt}$

# 3. Internal state matc
Code update:
- I rebuilt the project to focus on matching the induced internal state of the model at layer -1
- The model has started returning phrases of random characters with "cat" sprinkled somewhere in there again (I turned off the banned words list)
- With it on, I get very light activations on neuronpedia, but still gibberish outputs

I think it may be time to look at other layers, going to look at the first layer, and some random internal layers too

| Layer | Step stopped at | Alternative prompt        | Sample continuation (first few tokens after prompt)                                                                             |
| ----- | --------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 0     | 44              | `Talk about cats.`        | `Cats are amazing creatures! 😻 They're known for their independence, grace, and playful nature.`                               |
| 2     | 112             | `Talk about cats.`        | `Do you have any? I've always been fascinated by cats and have a soft spot for them.`                                           |
| 4     | 120             | `Talk about cats.`        | `They can be so mysterious and independent creatures, but they also seem to have secrets...`                                    |
| 6     | 112             | `Talk about cats.`        | `I know a lot of people who love cats... are cats really as aloof as the stories say?`                                          |
| 8     | 111             | `Talk about cats im`      | `sure you will agree that they are the queens of all things cool. I was at the park today...`                                   |
| 10    | 169             | ` quand about cats.`      | `**This post focuses on the positive aspects of being a cat owner.** There's an emphasis on the joys...`                        |
| 12    | 125             | `Reports ABOUT cats.`     | `Here are a few: **Cats that can't stop purring.** It sounds like they have a good time! 😸`                                    |
| 14    | 499             | ` about catsAbout`        | `**Cats, with their independent nature and quirky behaviors, have captivated humans for centuries.**`                           |
| 16    | 499             | `Tried</ About cats公民`    | `I like cats. I think they're really cool! But I also hate cats that are shedding hair all over...`                             |
| 18    | 499             | `Talk about cats pickups` | `...Well, I've been thinking a lot lately about guitars and their sound. I love the crunchy tones...`                           |
| 20    | 499             | `orre<h3> aboutcats">`    | `<span class="link">Click here</span> </div> ``` The HTML code you provided lacks a closing </div> tag...`                      |
| 22    | 499             | ` scarcely)ard cats Bor`  | `rowing from the language of their prey, these small predators have evolved to become some of nature's most skilled hunters...` |
| 24    | 499             | `跌 graph About cats বল`   | `লেন। The graph shows the average daily income of cats in different countries...`                                               |
|       |                 |                           |                                                                                                                                 |
I've tunes the algorithms somewhat, notably:
- Increasing the training steps to 1000 to allow the prompt more time to converge
- Increasing the banned token list to include $\{s,\text{caps}(S),\text{begin\_caps}(S) \forall s \in \{\text{"talk", "about", "cats"}\}^*+\text{.|" "}\}$
- Increased the banned word weight to `1`
and...

| Layer | Step stopped at | Alternative prompt                  | Sample continuation (first few tokens after prompt)                                                                                                                              |
| ----- | --------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | 999             | `Talk about cats.`                  | Cats are amazing creatures! 😻 They're known for their independence, grace, and playful nature.                                                                                  |
| 2     | 999             | `Talk about cats।`                  | Cats are fascinating creatures that have captivated human hearts for centuries.                                                                                                  |
| 4     | 999             | `Talk about cats.\r`                | I've always been fascinated by them. They seem so independent and yet so loving at the same time.                                                                                |
| 6     | 999             | `Talk about catstoxicity`           | I am writing a story about a feline who had to deal with the chaos that emerged when cats discovered catsup as an edible item.                                                   |
| 8     | 985             | `Talkabout catessori`               | Catnip toys are a hit, but did you know about the amazing benefits of other Montessori-inspired toys for your kitty?                                                             |
| 10    | 617             | `Answerabout catsн`                 | Cat behavior can be a fascinating study in contradictions, often stemming from their independent nature and strong instincts honed over millennia as predators.                  |
| 12    | 958             | `BahЧто about cats’.`               | Please provide more context: what kind of information are you looking for regarding cats?                                                                                        |
| 14    | 999             | `borg about catsespacio`            | Catspace: your personal cat-friendly hotel room where your furry friend can lounge in luxury and enjoy an enriching experience.                                                  |
| 16    | 986             | `graphics lee ancatsu`              | I like the name "Lee Ancatsu." It has a nice ring to it and sounds good!                                                                                                         |
| 18    | 999             | `Entonces /></ tentang cats choses` | Si vous avez besoin d'aide pour choisir les meilleurs choix de repos pour votre chat, n'hésitez pas à me contacter.                                                              |
| 20    | 999             | `Lo thuyAboutCat丁目`                 | The increasing prevalence of AI adoption sparks debate on the future of human existence and raises concerns about job displacement, ethical considerations, and potential risks. |
| 22    | 999             | `ЛоBad jak Catsjú`                  | "I'm looking for an escape room," said the woman, her eyes wide with anticipation, "but I've heard there's a lot of competition."                                                |
| 24    | 999             | `Allianceوندunecats排水`              | 2015年，该公司收购了美国一家名为“The Drain Guys”的排水管清洗服务公司，进一步扩展了业务。                                                                                                                           |

# 4. Prefix-suffix generation matching
Since the end-goal of this is to generate a "steering prefix" $P_s$ so that I can encourage the model to talk about something without explicitly telling it to in english, and $P_s$ is used in combination with some suffix $P_z$, I should be trying to optimize $P_s$ *in-context* (ie. look at the final internal states when the model is given $P_s+P_z$)

I've built the steering prompt generator now with `P_REF` being "Talk about cats" and using greedy decoding to get rid of nondeterminism in the model's outputs

ChatGPT also recommended using a teacher-forcing dataset
- How to score a prompt by how well it predicts a target answer
- Uses cross-entropy loss between the desired reference output (`Y_REF`) and the generated output

Uses nearest-neighbor w/ cosine similarity to project soft prompt to real tokens from the embedding matrix
- Nearest neighbors in the embedding matrix (closest embeddings in meaning to generated output)

Making sure that the reference is decently different from the reference

>[!NOTE]
>I'm running the whole algorithm several times to see if I can randomly get a better output

The distribution for that (8 samples) looks like this:
![[Pasted image 20251219184647.png]]
*I'd like to run this a few times to see what the effectiveness of this approach is*
- Maybe compare it to the number of training steps it took to get here

Improves the projected token with HotFlip-esque coordinate descent
- Gradients are W.R.T prefix posiitons
- Tokens are replaced that improve CE while still being dissimilar from the original prompt

Prompt is evaluated with:
- Teacher-forced CE
- Exact match rate
- Exact match rate for the first k tokens (The model will probably diverge)
- Overall token match rate

I got a bunch of prompts that seem to work pretty well! Some examples:
- These are generated with the prompt "$P+S$", where $P$ is some prefix, $S$ is some suffix, $P_{ref}$ is the reference prefix (ie. "Talk about cats"), and $P_{cand}$ is the generated prefix
- Best $P_{cand}$ is `'〮 درباره cats.テナンスhopped⊐.gnition\ue806 cats opportunity ſindamerikanische $_=`
- Candidate metrics: `CE: 0.6109491586685181 Exact: 0.0 Exact@K: 0.0 Tok@K: 0.10723860589812333`

| Similarity             | SUFFIX                                                        | SAME?   | REF (first sentence-ish)                                                                | CAND (first sentence-ish)                                                                             |
| ---------------------- | ------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Most similar #1**    | Summarize the following text in one sentence, neutral tone.   | ✅ True  | “Please provide the text you would like me to summarize.”                               | “Please provide the text you would like me to summarize.”                                             |
| **Most similar #2**    | Respond with exactly one emoji.                               | ✅ True  | “🐈”                                                                                    | “🐈”                                                                                                  |
| **Most similar #3**    | Transform the text into a short rhyme of four lines.          | ❌ False | “With fur so soft and eyes so bright…”                                                  | “Cats are furry, sleek and sly…”                                                                      |
| **Most dissimilar #1** | Respond only with a single lowercase word.                    | ❌ False | “cats”                                                                                  | “love”                                                                                                |
| **Most dissimilar #2** | Provide a concise step-by-step plan with exactly three steps. | ❌ False | “## A Cat Owner’s Quick Guide: 1. **Provide a Safe Haven:** …”                          | “It seems like you’re asking for a plan about cats, but the provided text is a jumbled mix of words.” |
| **Most dissimilar #3** | Paraphrase using simpler language for a general audience.     | ❌ False | “Cats are amazing creatures! They’re known for being independent, playful, and cuddly.” | “Please provide me with the text you want me to paraphrase! I need the original text to help you.”    |

There's definitely a pattern in a lot of the responses, I've saved the entire output in [[prefix-suffix-optimizer-first-output]]

I accidentally ran the prompt optimizer without the banned token ids list instantiated (and prompt len = 16), and all the outputs were "Talk abut cats. Talk about cats....", although the model seemed to forego spaces between each sentence

I wanted to give the model a bit more leeway, so I extended the teacher prompt to "Talk only about cats. Do not talk about anything else."

Even with the banned tokens, the model is producing prompts with cats in it, and I got a prompt with "Talk only about cats" appended to it

>[!NOTE]
>This model *really* wants to have "talk about cats" in the output, and I've always had a hard time getting it to not have cats in the prompt

The soft prompt is actually really good, reliably getting a sub 0.0001 ce score, but when it's projected to tokens the best I've gotten is ~0.32
- My method of using hotflip nearest neighbors is interesting, but it only tests individual nearest tokens (which has never actually revealed a better token to use). It would be helpful to look at k or so tokens at a time, (ie. "Closest tok phrase" instead of "Closest token)
- I also may need to optimize the training based on discrete tokens instead of a soft prompt
	- This is gonna be so computaitonally expensive 
	- Again, already sort of doing this with hotflip but it's not quite good enough
	- The overall algorithm & idea works! It's just a matter of giving it the right input now

>[!NOTE]
>All of this is with greedy sampling, the concepts may differ vastly when changed to another sampling method (topk, temp, etc)

>[!NOTE]
> The $\bar{x}$ CE difference of 6.18624e-05 ($s = 5.92923e^{-21}$) could just be weird greedy decoding chaos
> - Small semantic differences in the logit scores could be the reason for this

Lots of random nasty bits in the generated prompts ("`juste9 試 weܞroupe⸨nuovoᅚ FacebookSign\ue386푅\ue368 ,\r monasteries snowboard`") would make it hard to use this in any sort of chat interface, since they're all directly decoded tokens and most interfaces treat inputs as text and decode them literally, might want to do a more strict ban if I wanted a live demo but this is a POC so it works for now

I've updated the suffixes list to contain less "Summarize the previous text" prompts (prompts that reference other text) since they tend to produce the same result no matter what the input was ("please provide the text you'd like me to summarize")

After playing with some params in hope the fix is easy (tweaking the length of the teacher prompt, first k tokens & weight, etc), the model is still getting really close with the soft prompts and then returning to the same CE when the soft prompt goes discrete. Going to implement a change to reconfigure the training algorithm to steer towards existing tokens instead of being allowed to fly off into uncharted logit space

OKAY WAIT
The ce is lowering AND IS BEING PRESERVED WHEN PROJECTED TO DISCRETE TOKENS
- *And the hotflip is actually running now!*

So it works well enough and I've tested this so much I think it's doing what I expect it to do, plus the model is finding ways to cheat my attempts at preventing it from realizing that "cat" and "Cat" are technically different tokens and therefore one is valid whereas the other is literally in the prompt

I do want to do a test of what pure hotflip does because this is what I'm seeing:
1. The soft prompt generator does its thing so well
2. The projection function destroys the ce and returns the ce to oblivion (makes it go waaay back up (~back to the pre-optimized random vector))
3. Hotflip restores it (sorta) to something reasonable (~.50)
Although I don't think this will work because:
- The soft prompt generator is sort of like a coarse tuning knob, finding the general prompt needed without searching through the entire embedding space every time for something that optimizes closeness to the teacher prompt
- Hotflip is a very fine tuning knob that looks for specific singular tokens that could be improved
In combination these work pretty well actually, but without the coarse knob the fine knob is useless (2304 embedding dimension, for one or two tokens, but even for 3 it's 12.23 billion combinations)
TLDR: My algorithm: coarse knob/general direction, HotFlip: fine-tuning knob


Curious to see if this technique could be applied with steering to find random equivalency steering directions or to see if we can find the "english engenvector", and then apply this and see if the model outputs "cat" in other languages


I tried a pure hotflip algorithm and I had a few issues:
- Initializing from a random prefix isn't the best because it starts so far away from the desired prompt
- Initializing from the desired prefix also isn't the best because that's the desired value, the single smallest step away from the desired prefix is also the single closest 

This shows in the training statistics, with an average CE of $0.8153$ across $n=5$

The combo approach seems to work a lot better a lot less consistently, whereas hotflip does pretty much the same most of the time

Forgot to say, the full prompt is calculated as:
$[chat\_pre] + prefix\_ids + [SEP] + suffix\_ids + [chat\_between] + completion\_ids + [chat\_post]$
And CE is the weighted average negative log-likelihood of the reference completion tokens, with the first EARLY_K completion tokens getting weight EARLY_WEIGHT, and later completion tokens getting weight 1. Non-completion tokens are ignored
- It may be beneficial to have a more continuous weight instead of an entirely discrete one

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
- *at the moment, $EARLY\_WEIGHT=3.0$ and $EARLY\_K=32$*

I think I'm ready to start on the writeup, I've found some really interesting stuff and I'm excited to do the write-up