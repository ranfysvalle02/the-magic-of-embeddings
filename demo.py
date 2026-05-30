"""
demo.py - The Magic of Embeddings, Demystified

A tiny, dependency-free walkthrough of the pipeline described in README.md:

    raw tokens -> embedding lookup (Box A) -> attention
               -> context vectors -> output weights -> logits

We use 4-dimensional, hand-authored embeddings so every step prints out as
readable numbers. The point is not to be realistic; it is to make the
mechanism legible.

Run with:
    python demo.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Tiny vector / matrix helpers (the whole math kit)
# ---------------------------------------------------------------------------

Vector = List[float]
Matrix = List[List[float]]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def scale(v: Vector, s: float) -> Vector:
    return [x * s for x in v]


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def matvec(m: Matrix, v: Vector) -> Vector:
    return [dot(row, v) for row in m]


def softmax(scores: List[float]) -> List[float]:
    # Subtract the max for numerical stability; mathematically identical.
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def fmt_vec(v: Vector, dims: List[str]) -> str:
    return "[" + ", ".join(f"{d}={x:+.2f}" for d, x in zip(dims, v)) + "]"


# ---------------------------------------------------------------------------
# Box A: the embedding LOOKUP TABLE
#
# This is literally a dict from token -> row of a matrix. Not a model. Not a
# transformer. Just a table that the rest of the network sits on top of.
#
# Hand-picked dimensions so the demo stays interpretable:
#   nature  - tree-ish / plant-ish stuff
#   canine  - dog-ish stuff
#   sound   - audio / acoustic stuff
#   texture - bark / wood / skin texture
# ---------------------------------------------------------------------------

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


def embed(tokens: List[str]) -> List[Vector]:
    return [EMBEDDING_TABLE[t] for t in tokens]


# ---------------------------------------------------------------------------
# "Learned" projections + output head
#
# In a real transformer, W_Q, W_K, W_V are learned matrices and Q != K != V
# even for the same token. To keep this demo legible we set them to identity,
# which makes attention scores reduce to plain dot products between embeddings.
# That is the clearest possible way to see "tokens pulling in meaning from
# the neighbors that matter to them".
# ---------------------------------------------------------------------------

def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


W_Q: Matrix = identity(len(DIMS))
W_K: Matrix = identity(len(DIMS))
W_V: Matrix = identity(len(DIMS))


# ---------------------------------------------------------------------------
# Lock-and-key projections: an example where Q != K for the same token
#
# These are *also* hand-authored, but instead of identity they implement a
# specific routing rule that the README hints at: the query a token sends
# out and the key it advertises are LEARNED LINEAR MAPS of its embedding,
# and they do not have to read the same dimensions.
#
# Reading the rows top-to-bottom:
#
#   W_Q_LOCKKEY (what a token ASKS FOR)
#     axis 0: my 'sound' is looking for someone canine
#     axis 1: vanilla self-channel on 'nature'
#     axis 2: vanilla self-channel on 'texture'
#     axis 3: unused
#
#   W_K_LOCKKEY (what a token ADVERTISES)
#     axis 0: I AM canine
#     axis 1: I AM nature
#     axis 2: I AM texture
#     axis 3: unused
#
# Axis 0 is where the asymmetric routing lives: a 'sound' query collides
# with a 'canine' key, so high-sound tokens (like 'loud') reach for
# high-canine tokens (like 'dog'), even though sound and canine are
# different dimensions of the embedding space.
#
# In real transformers Q-dim and K-dim are often smaller than D-dim (one
# per attention head). We keep them 4x4 here only to fit the pattern.
# ---------------------------------------------------------------------------

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


# A tiny "output head". Each row is a direction that reads a context vector
# and produces a logit for one meaning label. We are not predicting next
# tokens here; we are predicting which sense of 'bark' is active.
OUTPUT_LABELS = ["tree_meaning", "dog_meaning"]
OUTPUT_WEIGHTS: Matrix = [
    [ 1.0, -0.5,  0.0,  1.0],   # tree_meaning loves nature + texture
    [-0.5,  1.0,  0.8, -0.2],   # dog_meaning  loves canine + sound
]


# ---------------------------------------------------------------------------
# Attention (a single head, no scaling, no masking)
# ---------------------------------------------------------------------------

def attention(
    tokens: List[str],
    w_q: Matrix = None,  # type: ignore[assignment]
    w_k: Matrix = None,  # type: ignore[assignment]
    w_v: Matrix = None,  # type: ignore[assignment]
) -> Tuple[List[Vector], List[List[float]]]:
    """Run one self-attention pass.

    The projections default to the module-level identity matrices, so calling
    attention(tokens) reproduces the original baseline. Pass alternative
    matrices to demo what changes when Q and K start asking different
    questions of the same token.

    Returns (context_vectors, attention_weights) where attention_weights[i]
    is the distribution token i used to mix the other tokens' values.
    """
    w_q = W_Q if w_q is None else w_q
    w_k = W_K if w_k is None else w_k
    w_v = W_V if w_v is None else w_v

    embeddings = embed(tokens)
    queries = [matvec(w_q, e) for e in embeddings]
    keys    = [matvec(w_k, e) for e in embeddings]
    values  = [matvec(w_v, e) for e in embeddings]
    value_dim = len(values[0])

    context_vectors: List[Vector] = []
    attn_rows: List[List[float]] = []
    for q in queries:
        scores = [dot(q, k) for k in keys]
        weights = softmax(scores)
        attn_rows.append(weights)
        ctx: Vector = [0.0] * value_dim
        for w, v in zip(weights, values):
            ctx = add(ctx, scale(v, w))
        context_vectors.append(ctx)
    return context_vectors, attn_rows


# ---------------------------------------------------------------------------
# Output head: read the context vector, never the raw token
# ---------------------------------------------------------------------------

def score_meaning(context_vector: Vector) -> Tuple[List[float], List[float]]:
    logits = matvec(OUTPUT_WEIGHTS, context_vector)
    probs = softmax(logits)
    return logits, probs


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def hr(title: str = "") -> None:
    bar = "=" * 72
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def print_attention_table(tokens: List[str], attn: List[List[float]]) -> None:
    col_w = max(6, max(len(t) for t in tokens))
    header = " " * (col_w + 2) + "  ".join(f"{t:>{col_w}}" for t in tokens)
    print(header)
    for tok, row in zip(tokens, attn):
        cells = "  ".join(f"{w:{col_w}.2f}" for w in row)
        print(f"  {tok:>{col_w}}  {cells}")


def walkthrough(
    sentence: str,
    w_q: Matrix = None,  # type: ignore[assignment]
    w_k: Matrix = None,  # type: ignore[assignment]
    w_v: Matrix = None,  # type: ignore[assignment]
    note: str = "",
) -> Tuple[Vector, List[List[float]], List[str]]:
    tokens = sentence.split()
    header = f"Sentence: \"{sentence}\""
    if note:
        header += f"   [{note}]"
    hr(header)

    print("\nStatic embeddings (Box A - just a table lookup, context-FREE):")
    for t, v in zip(tokens, embed(tokens)):
        print(f"  {t:>6} -> {fmt_vec(v, DIMS)}")

    ctxs, attn = attention(tokens, w_q=w_q, w_k=w_k, w_v=w_v)

    print("\nAttention weights (rows = query token, cols = key token):")
    print_attention_table(tokens, attn)

    bark_idx = tokens.index("bark")
    print("\nThe 'bark' row above is the briefing memo's recipe.")
    print("Mixing the value vectors with those weights produces the")
    print("context-AWARE vector for 'bark':")
    print(f"  static  bark -> {fmt_vec(EMBEDDING_TABLE['bark'], DIMS)}")
    print(f"  context bark -> {fmt_vec(ctxs[bark_idx], DIMS)}")

    logits, probs = score_meaning(ctxs[bark_idx])
    print("\nOutput head reads the CONTEXT vector (never the raw word):")
    for label, lg, p in zip(OUTPUT_LABELS, logits, probs):
        print(f"  {label:>14}  logit={lg:+.3f}   prob={p:.3f}")

    winner = OUTPUT_LABELS[probs.index(max(probs))]
    print(f"  -> winning sense: {winner}")
    return ctxs[bark_idx], attn, tokens


# ---------------------------------------------------------------------------
# README concept callouts
# ---------------------------------------------------------------------------

def extension_lock_and_key(
    sentence: str,
    baseline_tokens: List[str],
    baseline_attn: List[List[float]],
) -> None:
    """Re-run a sentence with non-identity Q/K projections.

    Shows the lock-and-key effect: a 'sound' query reaches across the
    sentence and grabs onto a 'canine' key, even though sound and canine
    are different embedding dimensions. This is the routing behaviour the
    README hints at when it says W_Q and W_K decide which properties
    become a 'search request' versus an 'advertised label'.
    """
    hr("Extension: what if Q and K read different dimensions?")
    print(
        "  Up to here we used W_Q = W_K = I, so a token's query and its\n"
        "  key were the same vector. That hides one of attention's most\n"
        "  important moves: Q and K are LEARNED LINEAR MAPS, and they do\n"
        "  not have to read the same dimensions of the embedding.\n"
        "\n"
        "  W_Q_LOCKKEY rewires channel 0 to: my 'sound' axis -> a query\n"
        "              that asks 'who here is canine?'\n"
        "  W_K_LOCKKEY rewires channel 0 to: my 'canine' axis -> a key\n"
        "              that advertises 'I am canine.'\n"
        "\n"
        "  Result: sound-of-self collides with canine-of-other on the\n"
        "  shared axis. 'loud' (high sound) should now reach for 'dog'\n"
        "  (high canine) much more strongly than before."
    )

    _, lockkey_attn, lockkey_tokens = walkthrough(
        sentence,
        w_q=W_Q_LOCKKEY,
        w_k=W_K_LOCKKEY,
        note="lock-and-key W_Q, W_K   (W_V still identity)",
    )

    # Focused side-by-side: 'loud' attending under each regime.
    assert lockkey_tokens == baseline_tokens, "token order must match for comparison"
    loud_idx = baseline_tokens.index("loud")
    dog_idx = baseline_tokens.index("dog")

    print("\nFocused comparison - 'loud' row, identity vs lock-and-key:")
    col_w = max(6, max(len(t) for t in baseline_tokens))
    print(" " * (10 + col_w + 2) + "  ".join(f"{t:>{col_w}}" for t in baseline_tokens))
    print(f"  {'identity':>{8 + col_w}}  "
          + "  ".join(f"{w:{col_w}.2f}" for w in baseline_attn[loud_idx]))
    print(f"  {'lock&key':>{8 + col_w}}  "
          + "  ".join(f"{w:{col_w}.2f}" for w in lockkey_attn[loud_idx]))

    base_dog = baseline_attn[loud_idx][dog_idx]
    lk_dog = lockkey_attn[loud_idx][dog_idx]
    delta = lk_dog - base_dog
    print(
        f"\n  loud -> dog jumped from {base_dog:.2f} to {lk_dog:.2f}  "
        f"(delta = {delta:+.2f})"
    )
    print(
        "  That delta is the lock-and-key in action: nothing about the\n"
        "  word 'loud' changed - we only changed the MATRICES that\n"
        "  decide which property of 'loud' becomes a search request.\n"
        "  Q and K can now ask different questions of the same token."
    )


def explain_box_a_vs_box_b() -> None:
    hr("Box A vs Box B - 'embedding' means two different things")
    print(
        "  Box A - the embedding LAYER inside this demo:\n"
        "    EMBEDDING_TABLE is literally a Python dict. No attention,\n"
        "    no QKV, no layers. token id -> row of a matrix. It is the\n"
        "    floor of the stack and feeds attention from below.\n"
        "\n"
        "  Box B - a standalone embedding MODEL (Voyage, BERT, ...):\n"
        "    A whole transformer. It uses attention internally, then\n"
        "    pools everything into ONE vector for an entire passage.\n"
        "    Its own first layer is its own Box-A lookup, so the\n"
        "    recursion bottoms out at a matrix - not 'transformers all\n"
        "    the way down'."
    )


def explain_training_vs_inference() -> None:
    hr("Training vs Inference - same forward pass, different freedoms")
    print(
        "  We just ran the forward pass: embed -> attend -> decide.\n"
        "  That pass is IDENTICAL in training and inference.\n"
        "\n"
        "  Inference (what this script does):\n"
        "    EMBEDDING_TABLE, W_Q / W_K / W_V, OUTPUT_WEIGHTS are FROZEN.\n"
        "    Run forward once, read the answer.\n"
        "\n"
        "  Training (what we are NOT doing here):\n"
        "    Run the same forward pass, compare to a target, then use\n"
        "    gradient descent to push on ALL of those knobs together -\n"
        "    including the embedding table and the attention projections.\n"
        "    The output head only ever learns to read context vectors,\n"
        "    which is why inference-time memos are written in exactly\n"
        "    the language the head was trained to interpret."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    hr("The Magic of Embeddings - a tiny, legible walkthrough")
    print(
        "  Pipeline (see README.md):\n"
        "    raw tokens -> embedding lookup (Box A) -> attention\n"
        "               -> context vectors -> output weights -> logits\n"
        "\n"
        "  The static embedding for 'bark' is identical in both sentences;\n"
        "  attention is what makes the difference."
    )
    print(f"\n  static bark -> {fmt_vec(EMBEDDING_TABLE['bark'], DIMS)}")

    tree_ctx, _, _ = walkthrough("the bark of the tree")
    dog_sentence = "the dog made a loud bark"
    dog_ctx, dog_attn, dog_tokens = walkthrough(dog_sentence)

    hr("Side by side: same word, two situational meanings")
    print(f"  static  bark        -> {fmt_vec(EMBEDDING_TABLE['bark'], DIMS)}")
    print(f"  context bark (tree) -> {fmt_vec(tree_ctx, DIMS)}")
    print(f"  context bark (dog)  -> {fmt_vec(dog_ctx,  DIMS)}")
    print(
        "\n  'bark' went in ambiguous and came out disambiguated.\n"
        "  Attention is the layer where context becomes geometry."
    )

    extension_lock_and_key(dog_sentence, dog_tokens, dog_attn)

    explain_box_a_vs_box_b()
    explain_training_vs_inference()

    hr("Done")


if __name__ == "__main__":
    main()

"""
python3 demo.py

========================================================================
  The Magic of Embeddings - a tiny, legible walkthrough
========================================================================
  Pipeline (see README.md):
    raw tokens -> embedding lookup (Box A) -> attention
               -> context vectors -> output weights -> logits

  The static embedding for 'bark' is identical in both sentences;
  attention is what makes the difference.

  static bark -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]

========================================================================
  Sentence: "the bark of the tree"
========================================================================

Static embeddings (Box A - just a table lookup, context-FREE):
     the -> [nature=+0.10, canine=+0.10, sound=+0.10, texture=+0.10]
    bark -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]
      of -> [nature=+0.10, canine=+0.10, sound=+0.10, texture=+0.10]
     the -> [nature=+0.10, canine=+0.10, sound=+0.10, texture=+0.10]
    tree -> [nature=+1.00, canine=+0.00, sound=+0.10, texture=+0.80]

Attention weights (rows = query token, cols = key token):
           the    bark      of     the    tree
     the    0.19    0.22    0.19    0.19    0.22
    bark    0.12    0.36    0.12    0.12    0.28
      of    0.19    0.22    0.19    0.19    0.22
     the    0.19    0.22    0.19    0.19    0.22
    tree    0.10    0.25    0.10    0.10    0.44

The 'bark' row above is the briefing memo's recipe.
Mixing the value vectors with those weights produces the
context-AWARE vector for 'bark':
  static  bark -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]
  context bark -> [nature=+0.57, canine=+0.29, sound=+0.21, texture=+0.40]

Output head reads the CONTEXT vector (never the raw word):
    tree_meaning  logit=+0.827   prob=0.677
     dog_meaning  logit=+0.088   prob=0.323
  -> winning sense: tree_meaning

========================================================================
  Sentence: "the dog made a loud bark"
========================================================================

Static embeddings (Box A - just a table lookup, context-FREE):
     the -> [nature=+0.10, canine=+0.10, sound=+0.10, texture=+0.10]
     dog -> [nature=+0.10, canine=+1.00, sound=+0.60, texture=+0.20]
    made -> [nature=+0.10, canine=+0.20, sound=+0.40, texture=+0.10]
       a -> [nature=+0.10, canine=+0.10, sound=+0.10, texture=+0.10]
    loud -> [nature=+0.00, canine=+0.20, sound=+1.00, texture=+0.00]
    bark -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]

Attention weights (rows = query token, cols = key token):
           the     dog    made       a    loud    bark
     the    0.15    0.18    0.16    0.15    0.17    0.18
     dog    0.09    0.31    0.12    0.09    0.17    0.22
    made    0.13    0.20    0.15    0.13    0.19    0.19
       a    0.15    0.18    0.16    0.15    0.17    0.18
    loud    0.11    0.21    0.15    0.11    0.27    0.16
    bark    0.10    0.24    0.12    0.10    0.14    0.30

The 'bark' row above is the briefing memo's recipe.
Mixing the value vectors with those weights produces the
context-AWARE vector for 'bark':
  static  bark -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]
  context bark -> [nature=+0.26, canine=+0.52, sound=+0.47, texture=+0.20]

Output head reads the CONTEXT vector (never the raw word):
    tree_meaning  logit=+0.203   prob=0.372
     dog_meaning  logit=+0.726   prob=0.628
  -> winning sense: dog_meaning

========================================================================
  Side by side: same word, two situational meanings
========================================================================
  static  bark        -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]
  context bark (tree) -> [nature=+0.57, canine=+0.29, sound=+0.21, texture=+0.40]
  context bark (dog)  -> [nature=+0.26, canine=+0.52, sound=+0.47, texture=+0.20]

  'bark' went in ambiguous and came out disambiguated.
  Attention is the layer where context becomes geometry.

========================================================================
  Extension: what if Q and K read different dimensions?
========================================================================
  Up to here we used W_Q = W_K = I, so a token's query and its
  key were the same vector. That hides one of attention's most
  important moves: Q and K are LEARNED LINEAR MAPS, and they do
  not have to read the same dimensions of the embedding.

  W_Q_LOCKKEY rewires channel 0 to: my 'sound' axis -> a query
              that asks 'who here is canine?'
  W_K_LOCKKEY rewires channel 0 to: my 'canine' axis -> a key
              that advertises 'I am canine.'

  Result: sound-of-self collides with canine-of-other on the
  shared axis. 'loud' (high sound) should now reach for 'dog'
  (high canine) much more strongly than before.

========================================================================
  Sentence: "the dog made a loud bark"   [lock-and-key W_Q, W_K   (W_V still identity)]
========================================================================

Static embeddings (Box A - just a table lookup, context-FREE):
     the -> [nature=+0.10, canine=+0.10, sound=+0.10, texture=+0.10]
     dog -> [nature=+0.10, canine=+1.00, sound=+0.60, texture=+0.20]
    made -> [nature=+0.10, canine=+0.20, sound=+0.40, texture=+0.10]
       a -> [nature=+0.10, canine=+0.10, sound=+0.10, texture=+0.10]
    loud -> [nature=+0.00, canine=+0.20, sound=+1.00, texture=+0.00]
    bark -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]

Attention weights (rows = query token, cols = key token):
           the     dog    made       a    loud    bark
     the    0.16    0.18    0.16    0.16    0.16    0.19
     dog    0.13    0.23    0.14    0.13    0.14    0.22
    made    0.14    0.21    0.15    0.14    0.15    0.20
       a    0.16    0.18    0.16    0.16    0.16    0.19
    loud    0.12    0.29    0.13    0.12    0.13    0.21
    bark    0.13    0.20    0.14    0.13    0.12    0.29

The 'bark' row above is the briefing memo's recipe.
Mixing the value vectors with those weights produces the
context-AWARE vector for 'bark':
  static  bark -> [nature=+0.70, canine=+0.70, sound=+0.40, texture=+0.40]
  context bark -> [nature=+0.26, canine=+0.47, sound=+0.43, texture=+0.19]

Output head reads the CONTEXT vector (never the raw word):
    tree_meaning  logit=+0.216   prob=0.393
     dog_meaning  logit=+0.651   prob=0.607
  -> winning sense: dog_meaning

Focused comparison - 'loud' row, identity vs lock-and-key:
                     the     dog    made       a    loud    bark
        identity    0.11    0.21    0.15    0.11    0.27    0.16
        lock&key    0.12    0.29    0.13    0.12    0.13    0.21

  loud -> dog jumped from 0.21 to 0.29  (delta = +0.08)
  That delta is the lock-and-key in action: nothing about the
  word 'loud' changed - we only changed the MATRICES that
  decide which property of 'loud' becomes a search request.
  Q and K can now ask different questions of the same token.

========================================================================
  Box A vs Box B - 'embedding' means two different things
========================================================================
  Box A - the embedding LAYER inside this demo:
    EMBEDDING_TABLE is literally a Python dict. No attention,
    no QKV, no layers. token id -> row of a matrix. It is the
    floor of the stack and feeds attention from below.

  Box B - a standalone embedding MODEL (Voyage, BERT, ...):
    A whole transformer. It uses attention internally, then
    pools everything into ONE vector for an entire passage.
    Its own first layer is its own Box-A lookup, so the
    recursion bottoms out at a matrix - not 'transformers all
    the way down'.

========================================================================
  Training vs Inference - same forward pass, different freedoms
========================================================================
  We just ran the forward pass: embed -> attend -> decide.
  That pass is IDENTICAL in training and inference.

  Inference (what this script does):
    EMBEDDING_TABLE, W_Q / W_K / W_V, OUTPUT_WEIGHTS are FROZEN.
    Run forward once, read the answer.

  Training (what we are NOT doing here):
    Run the same forward pass, compare to a target, then use
    gradient descent to push on ALL of those knobs together -
    including the embedding table and the attention projections.
    The output head only ever learns to read context vectors,
    which is why inference-time memos are written in exactly
    the language the head was trained to interpret.

========================================================================
  Done
========================================================================
"""