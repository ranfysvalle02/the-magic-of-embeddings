# the-magic-of-embeddings

---

# Attention, Decoded: How Embeddings, Training, and Inference Actually Fit Together

Large language models can feel like sorcery: you type a prompt, and a few seconds later a context-aware answer appears, as if something on the other side genuinely *understood* you. But under the hood there's no ghost in the machine — just a beautifully coordinated pipeline of geometry and statistics, with one mechanism doing the heavy lifting: **attention**.

To talk about it without drowning in math, we'll lean on a single analogy throughout. Picture a brilliant but impossibly busy **CEO** who never has time to read the raw, messy stack of reports landing on their desk. Instead they rely on an **Expert Research Assistant** who scans everything, filters the noise, and hands over a tight **1-Page Briefing Memo**. The CEO reads only the memo, then makes the call. Hold that image — by the end you'll see it *is* the architecture, almost line for line:

- The **CEO's instincts** are the model's learned weights.
- The **Research Assistant** is attention.
- The **Briefing Memo** is the context vector that bridges the two.

Even with that picture in hand, most people run into the same wall — a nagging set of questions that make the whole thing feel slippery:

- Where exactly does **attention** sit? Is it the start of the pipeline, the middle, or a trick layered on at the end?
- What is its relationship to **embeddings** — and wait, don't you need an *embedding model* to get embeddings? Isn't *that* a transformer too? Is it transformers all the way down?
- Does attention behave differently during **training** versus **inference**, or is it the same machine running in two modes?

This post exists to dissolve that confusion completely. By the end, you'll have one clean mental model where embeddings, attention, training, and inference each have exactly one job and one place to stand.

And because hand-waving is the enemy of understanding, every claim that follows is backed by [`demo.py`](demo.py) — a tiny, dependency-free Python script that prints the numbers as they pop out of each stage. Run it with `python demo.py` while you read; every block of output you see in this post comes from there.

---

## The One-Sentence Role of Attention

Let's start with the thesis and spend the rest of the post earning it:

> **Attention is the layer that turns a pile of independent, context-free word coordinates (embeddings) into context-aware representations — by letting every token pull in meaning from the other tokens that matter to it.**

Everything else is a consequence of that single idea:

- **Embeddings** give you *static* meaning ("bark" has both tree-ish and dog-ish properties, always).
- **Attention** gives you *situational* meaning ("in *this* sentence, bark means tree").
- **The output weights** read that situational meaning to make a decision (predict the next word).

Attention is the hinge in the middle. It is not the input, and it is not the final decision. It is the **representation layer** that connects them.

---

## The Vertical Stack: Where Attention Actually Sits

Here is the single most important diagram to burn into memory. A transformer is a strict, bottom-to-top pipeline:

```
[ Raw tokens ]            "the", "bark", "of", "the", "tree"
       │  (lookup)
       ▼
[ Embeddings ]            a static coordinate per token — context-FREE
       │  (Q, K, V projections + attention)
       ▼
[ Context vectors ]       the same tokens, now context-AWARE  ← attention's output
       │  (more layers... then the output weights)
       ▼
[ Logits → Softmax ]      a probability distribution over the next token
```

Map this onto our corporate analogy:

- **Embeddings** are the raw files landing on the desk.
- **The context vector** is the briefing memo.
- **Attention** is the assistant who reads the files and writes the memo.
- **The output weights** are the CEO's instincts, reading only the memo.

The thing to internalize: **attention is a *transformation between two representations*, not a module bolted onto the side.** It takes embeddings in and hands context vectors out. That's its whole life.

To make that absolutely concrete: in [`demo.py`](demo.py) the bottom of the stack is literally a Python dict. Four hand-picked dimensions — `nature`, `canine`, `sound`, `texture` — and a row per word.

```72:83:demo.py
DIMS = ["nature", "canine", "sound", "texture"]

EMBEDDING_TABLE: Dict[str, Vector] = {
    "the":  [0.10, 0.10, 0.10, 0.10],
    "of":   [0.10, 0.10, 0.10, 0.10],
    "a":    [0.10, 0.10, 0.10, 0.10],
    "made": [0.10, 0.20, 0.40, 0.10],
    "bark": [0.70, 0.70, 0.40, 0.40],  # ambiguous on purpose
    "tree": [1.00, 0.00, 0.10, 0.80],
    "dog":  [0.10, 1.00, 0.60, 0.20],
    "loud": [0.00, 0.20, 1.00, 0.00],
}
```

That dict is Box A. `embed(tokens)` is just `[EMBEDDING_TABLE[t] for t in tokens]`. Notice `bark` has both nature-ish *and* canine-ish properties baked in — it sits on the bridge between two neighbourhoods, exactly as advertised. The whole rest of the post is about what attention does with that ambiguous row once it's looked up.

---

## Attention's Relationship to Embeddings

This is the relationship people most often get backwards, so let's be exact about it.

**1. Embeddings are the *input* to attention, and they are context-free.**

The vector for "bark" is identical whether the sentence is about a tree or a dog. It simply encodes "this word has both nature-ish and canine-ish properties." It sits on a *bridge* between two neighborhoods in semantic space, holding properties of both, committing to neither.

**2. Attention's job is to *resolve* that ambiguity using the neighbors that actually showed up.**

It produces a *new* vector for "bark" — one that has been nudged hard toward "nature" because the word "tree" was present in the sentence. The static embedding went in ambiguous; the context vector comes out disambiguated. The word has been mathematically updated to mean *tree bark* before it ever reaches the output weights.

**3. The crucial subtlety: the Q/K/V matrices operate *on top of* the embedding space.**

Attention never touches raw words. It operates on embeddings, and the Query, Key, and Value projections are *learned linear maps of those embeddings*. The query that "bark" sends out is literally:

```
query(bark) = W_Q × embedding(bark)
```

So embeddings and attention aren't two separate "topics" you study independently. **Attention is *defined on top of* the embedding space.** The embedding holds the latent properties; the learned projections (`W_Q`, `W_K`) decide which of those properties become a *search request* versus an *advertised label*. That distinction is the entire reason a Query and a Key can differ for the same word.

The clean summary:

> **Embedding = the address. Attention = the routing logic that reads addresses and rewrites them based on the neighborhood that actually turned up.**

And to prove the address-rewriting isn't a metaphor, here's what falls out of [`demo.py`](demo.py) when you run the two sentences. The static row for `bark` is identical in both. The context vector that comes out of attention is not.

```
static  bark        -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]
context bark (tree) -> [nature=+0.57, canine=+0.29, sound=+0.21, texture=+0.40]
context bark (dog)  -> [nature=+0.26, canine=+0.52, sound=+0.47, texture=+0.20]
```

Same word, two situational meanings. Canine collapses in the tree sentence; nature collapses in the dog sentence. The output head — a tiny matrix scoring `tree_meaning` vs `dog_meaning` from the context vector alone — reads `tree_meaning` at 0.68 for the first and `dog_meaning` at 0.63 for the second, never seeing the raw word at all.

---

## Lock and Key: Why Q and K Aren't the Same Vector

The cleanest version of the demo cheats slightly. It uses `W_Q = W_K = W_V = I`, which collapses queries, keys, and values back into the original embedding. That makes the dot products easy to read but hides the whole reason Q and K exist as separate matrices in the first place.

So in [`demo.py`](demo.py) there's a second pair of projections, hand-authored to make the asymmetry visible:

```140:152:demo.py
W_Q_LOCKKEY: Matrix = [
    [0.0, 0.0, 1.0, 0.0],   # ask channel 0: my sound -> seeks canine
    [1.0, 0.0, 0.0, 0.0],   # ask channel 1: my nature -> seeks nature
    [0.0, 0.0, 0.0, 1.0],   # ask channel 2: my texture -> seeks texture
    [0.0, 0.0, 0.0, 0.0],
]

W_K_LOCKKEY: Matrix = [
    [0.0, 1.0, 0.0, 0.0],   # advertise channel 0: I am canine
    [1.0, 0.0, 0.0, 0.0],   # advertise channel 1: I am nature
    [0.0, 0.0, 0.0, 1.0],   # advertise channel 2: I am texture
    [0.0, 0.0, 0.0, 0.0],
]
```

Read channel 0 of each. The query side pulls from a token's *sound* dimension. The key side pulls from a token's *canine* dimension. They meet on the same axis. The geometric consequence: a high-sound query collides with a high-canine key — across tokens, on a dimension the two words don't even share inside the embedding.

Re-running `the dog made a loud bark` with these projections and comparing `loud`'s row of the attention matrix gives:

```
                 the     dog    made       a    loud    bark
    identity    0.11    0.21    0.15    0.11    0.27    0.16
    lock&key    0.12    0.29    0.13    0.12    0.13    0.21

loud -> dog jumped from 0.21 to 0.29  (delta = +0.08)
```

Under identity, `loud`'s strongest attention is on itself. Under lock-and-key, it's on `dog` — and nothing about the embedding for `loud` has changed. We only rewired which property of `loud` becomes a search request and which property of `dog` becomes an advertised label. That, finally, is the concrete payoff for the earlier claim that:

> The learned projections (`W_Q`, `W_K`) decide which of those properties become a *search request* versus an *advertised label*.

A query and a key can disagree about which dimension matters because they're different functions of the same row. That single fact is half of what makes attention more than a similarity search.

---

## The Confusing Part: "Don't You Need an Embedding Model?"

Here is the question that tangles everyone up:

> *"To get embeddings, you need an embedding model. And that model is itself transformer-powered, right? So it's transformers all the way down?"*

The confusion is real, but it comes from one word — **"embedding"** — being used for **two completely different things**. One of them is a transformer. The other absolutely is not. Put them in separate boxes and the recursion dissolves instantly.

### Box A: The Embedding *Layer* Inside a Transformer

This is the kind of embedding we've been talking about so far — the static coordinates that feed attention. It is **not** a transformer. It isn't even clever. It is a **lookup table** — one big matrix where row *N* is the vector for token *N*.

```
"bark"  ──(token id = 8423)──>  row 8423 of the matrix  ──>  [0.7 nature, 0.7 canine, ...]
```

That's the entire operation: an integer index into a table. No attention, no QKV, no layers. It is the *bottom rung* of the stack — it sits **below** attention and **feeds** it. In [`demo.py`](demo.py) this is laid bare: `EMBEDDING_TABLE` is a Python dict, and `embed()` is one line:

```86:87:demo.py
def embed(tokens: List[str]) -> List[Vector]:
    return [EMBEDDING_TABLE[t] for t in tokens]
```

There is no model under there. You hit the floor.

**Where do the table's numbers come from?** They're just weights. They start random and get sculpted by gradient descent during training, *jointly* with `W_Q/W_K/W_V` and the output weights — all in the same backward pass. No separate model produces them. The table *is* part of the transformer.

### Box B: A Standalone Embedding *Model*

*This* is the thing that's a full transformer. It uses attention internally, exactly like an LLM does. But its **job** is different:

- An **LLM** takes text and predicts the **next token** (it outputs a probability distribution over words).
- An **embedding model** takes text and outputs **one single vector** that summarizes the whole passage's meaning (it outputs coordinates, not words).

So the instinct "to get embeddings you need a transformer-powered model" is **correct — but only for Box B.** The document and sentence vectors used for search and retrieval come from a full transformer.

### Why It Isn't Infinite Recursion

Because Box B doesn't reach outside itself for embeddings. Every transformer carries its *own* internal Box-A lookup table as its first layer. The chain is finite and bottoms out:

```
Standalone embedding model (Box B = a transformer)
   └─ its FIRST layer is its own embedding lookup table (Box A: just a matrix)
        └─ then attention layers run on top
             └─ then it pools everything into ONE output vector
```

The lookup table is the **floor**. It bottoms out at "integer → row of a matrix." There is no model beneath it.

- To get *an LLM's internal* embeddings → **no model needed**, just a table lookup (Box A).
- To get *a document embedding for search* → **yes, a transformer** (Box B) — whose own first step is its own Box-A lookup.

### Hold It In Your Head Like This

| | Box A: embedding **layer** | Box B: embedding **model** |
|---|---|---|
| What is it? | A lookup table (one matrix) | A whole transformer |
| Uses attention? | No — it sits *below* attention | Yes — internally |
| Input | one token (an integer id) | a whole passage of text |
| Output | one vector **per token** | one vector for the **whole text** |
| Where it lives | the first layer of *every* transformer | a standalone model (Voyage, BERT, etc.) |
| Role | gives attention its raw material | produces vectors for search / RAG |

It's transformers all the way down **to a simple matrix** — not all the way down forever.

---

## Attention in Training vs. Inference

Now the last piece: does attention behave differently when the model is *learning* versus when it's *answering*?

Here is the insight that most explanations butcher:

> **Attention is the *exact same mechanism* in both phases. It is not an inference-only trick.**

The forward pass — embed, then attend, then decide — runs **identically** during training and inference. The difference between the two phases is *not what attention does*. The difference is whether the knobs are allowed to move.

| | Training | Inference |
|---|---|---|
| Embedding table | being learned / adjusted | frozen |
| `W_Q, W_K, W_V` (attention) | being learned / adjusted | frozen |
| Output weights | being learned / adjusted | frozen |
| Forward pass (embed → attend → decide) | **runs identically** | **runs identically** |
| Backward pass (gradient descent) | yes — updates all weights | no |

So, precisely:

- **Inference** = run the forward pass once with everything frozen. Attention builds the context vector; softmax picks a word.
- **Training** = run that *same* forward pass, measure the error, then use gradient descent to push on *every* learnable knob — **including the attention projections themselves.**

This is the correction that keeps the whole architectural picture from collapsing: **attention is not a passive observer during training. It is one of the main things being shaped.** Gradient descent doesn't only teach the CEO how to decide — it teaches the Assistant how to write better memos.

And this is why the output weights stay coherent: they *never* learn to read raw words. In **both** phases, they only ever see attention's output. The model learns to interpret exactly the kind of thing it will be handed at runtime. Training and inference share the same intermediate language — the context vector — which is the reason the model trained yesterday still makes sense of your prompt today.

[`demo.py`](demo.py) makes the "frozen" half of this picture very literal — `EMBEDDING_TABLE`, `W_Q`, `W_K`, `W_V`, `OUTPUT_WEIGHTS`, and `W_Q_LOCKKEY` / `W_K_LOCKKEY` are all module-level constants. The script only ever runs the forward pass. The "training" half of the table would simply replace each of those constants with a tensor that has a gradient — same code path, same dot products, same softmax — just with the additional rule that after every forward pass, every constant gets a small nudge in the direction that reduced the error. Nothing else changes.

> When the CEO went to business school, the Assistant sat right next to them the entire time. The CEO never learned to read raw files — they learned to read the Assistant's memos. So when a real memo lands on the desk at inference time, it's written in precisely the language the CEO spent years learning to interpret.

---

## The Complete Picture in Five Sentences

1. Embeddings give each token a fixed, context-free location in meaning-space (just a lookup table — Box A).
2. Attention, via learned Q/K/V projections, lets each token query its neighbors and rewrite itself into a context-aware vector — the briefing memo.
3. The output weights only ever read that context vector, never the raw tokens.
4. Inference runs this pipeline once with everything frozen; training runs the *identical* pipeline, then uses gradient descent to adjust the embedding table, the attention projections, *and* the output weights together.
5. So attention is the permanent **representation layer** that both *produces* runtime meaning **and** is itself *sculpted* during training — the hinge connecting embeddings (input) to decisions (output).

---

## So, Where Does the Magic Live?

Not in a ghost in the machine, and not in any single one of these parts. It lives in the **clean division of labor**:

- A dumb lookup table that knows static meaning.
- An attention layer that turns static meaning into situational meaning.
- An output layer that only ever reads situational meaning.
- A training process that sculpts all three at once, then freezes them.

Attention is the piece in the middle that makes the other three worth having. Embeddings would be inert without it; the output weights would be reading noise without it. It is the layer where context becomes geometry — and that, far more than any single learned fact, is the real magic of attention.

---

## Run it Yourself

Everything in this post is reproducible in under a second on any Python 3 install. No dependencies, no GPUs, no model downloads.

```
python demo.py
```

The script walks the pipeline end-to-end: it prints the static embeddings (Box A), the full attention matrix per sentence, the context vector for `bark` under each scenario, and the output head's verdict. It then re-runs the dog sentence with the lock-and-key projections so you can watch `loud`'s attention shift from itself to `dog` in real time.

If a paragraph in this post stopped making sense, find the matching section in [`demo.py`](demo.py), change a number, and run it again. The numbers don't lie — and once you've watched the same `bark` vector come out two different ways, the magic stops feeling like magic.
