"""Long-context retrieval and tracking, after RULER (Hsieh et al., 2024), scored by rule.

Five synthetic tasks at each context length, generated from a fixed seed so every model sees the same prompts:
  niah_single     one key in a long haystack; return its number
  niah_multikey   the same, among many look-alike keys
  niah_multivalue one key with four numbers scattered through the text; return all four
  vt              variable tracking: a chain of assignments hidden in noise; return every variable equal to a value
  cwe             common words: a long list of words; return the ten most frequent
Score per item = share of the expected strings found in the reply (RULER's string_match_all).
Lengths are approximate token counts (about 1.3 tokens per word, checked on Llama- and GLM-family tokenizers).

  inspect eval tasks/ruler.py -T lengths=32768,131072 -T per_task=8
"""
import random

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

TOK_PER_WORD = 1.3
NOISE = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
         "The river runs to the sea and the hills stand quietly around it. ").split(" ")
WORDS = ("amber basin cobalt delta ember fjord granite harbor island juniper kestrel lagoon meadow nectar orchard "
         "pebble quarry raven saddle timber umber valley willow yonder zephyr anchor bramble canyon dune estuary "
         "falcon glacier heron ivory jasper kelp lantern marble nimbus oasis prairie quill reef sierra tundra").split()


def words_for(tokens):
    return int(tokens / TOK_PER_WORD)


def haystack(rng, n_words, inserts):
    """noise text of n_words with each (position fraction, sentence) inserted"""
    base = [NOISE[i % len(NOISE)] for i in range(n_words)]
    for frac, sent in sorted(inserts, key=lambda x: -x[0]):
        base.insert(int(frac * len(base)), sent)
    return " ".join(base)


def key(rng):
    return "-".join(rng.choice(WORDS) for _ in range(3))


def num(rng):
    return str(rng.randint(1_000_000, 9_999_999))


def niah_single(rng, n):
    k, v = key(rng), num(rng)
    text = haystack(rng, n, [(rng.random(), f"One of the special magic numbers for {k} is: {v}.")])
    return f"{text}\n\nWhat is the special magic number for {k} mentioned in the text above? Answer with the number.", [v]


def niah_multikey(rng, n):
    keys = list({key(rng) for _ in range(40)})
    inserts, ans = [], {}
    for k in keys:
        ans[k] = num(rng)
        inserts.append((rng.random(), f"One of the special magic numbers for {k} is: {ans[k]}."))
    k = rng.choice(keys)
    return (f"{haystack(rng, n, inserts)}\n\nWhat is the special magic number for {k} mentioned in the text above? "
            "Answer with the number."), [ans[k]]


def niah_multivalue(rng, n):
    k, vals = key(rng), [num(rng) for _ in range(4)]
    text = haystack(rng, n, [(rng.random(), f"One of the special magic numbers for {k} is: {v}.") for v in vals])
    return f"{text}\n\nWhat are all the special magic numbers for {k} mentioned in the text above? List all four.", vals


def vt(rng, n):
    names = rng.sample([f"{a}{b}{c}" for a in "QWERTYUP" for b in "ASDFGHJK" for c in "ZXCVBNML"], 10)
    v = str(rng.randint(10000, 99999))
    chain = [f"VAR {names[0]} = {v}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, 5)]
    # a second chain as a distractor
    v2 = str(rng.randint(10000, 99999))
    chain2 = [f"VAR {names[5]} = {v2}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(6, 10)]
    fr = sorted(rng.random() for _ in range(5))
    fr2 = sorted(rng.random() for _ in range(5))
    text = haystack(rng, n, list(zip(fr, chain)) + list(zip(fr2, chain2)))
    return (f"{text}\n\nFind all variables that are assigned the value {v} in the text above, directly or through "
            "other variables. List their names."), names[:5]


def cwe(rng, n):
    vocab = rng.sample(WORDS, 40)
    top, rest = vocab[:10], vocab[10:]
    seq = []
    while len(seq) < n:
        seq += top * 3 + rng.sample(rest, 20)  # top words 3x as frequent as any other
    seq = seq[:n]
    rng.shuffle(seq)
    lst = " ".join(f"{i + 1}. {w}" for i, w in enumerate(seq))
    return (f"Below is a numbered list of words.\n{lst}\n\nWhat are the ten most common words in the list above? "
            "Answer with the ten words."), top


TASKS = {"niah_single": niah_single, "niah_multikey": niah_multikey, "niah_multivalue": niah_multivalue,
         "vt": vt, "cwe": cwe}


@scorer(metrics=[mean(), stderr()])
def string_match_all():
    async def score(state: TaskState, target: Target) -> Score:
        out = (state.output.completion or "").lower()
        hit = [t for t in target.target if t.lower() in out]
        return Score(value=len(hit) / len(target.target), answer=out[-300:])
    return score


@task
def ruler(lengths: str | list = "32768,131072", per_task: int = 8, seed: int = 2026):
    rng = random.Random(seed)
    samples = []
    # -T lengths=32768,131072 arrives as a list from the command line
    for L in [int(x) for x in (lengths if isinstance(lengths, list) else str(lengths).split(","))]:
        for name, fn in TASKS.items():
            # cwe: numbered entries run about 5.6 tokens each; capped at 32K (a counting test over the whole list)
            n = int(min(L, 32768) / 5.6) if name == "cwe" else words_for(L) - 120
            for i in range(per_task):
                prompt, target = fn(rng, n)
                samples.append(Sample(id=f"{name}-{L}-{i}", input=prompt, target=target,
                                      metadata={"task": name, "length": L}))
    return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=string_match_all())
