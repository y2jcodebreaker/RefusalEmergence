"""A1 — how many refusal REGISTERS are there, and does the taxonomy replicate across families?

    python register_taxonomy.py --k 2,3,4,5,6      # explore
    python register_taxonomy.py --k 4 --report     # fit, name, write the JSON

THE QUESTION. Ablating the refusal direction leaves ~60% of refusals intact and shifts them
from a first-person register ("I can't help with that") to a third-person normative one
("that is illegal and unethical"). Two registers turned up by accident, from reading
disagreements. Is that a dichotomy, or a slice of something larger?

THE TRAP, NAMED FIRST. It is trivially easy to invent categories and then find them: write a
phrase list, match it, report the categories the list encodes. That is not a taxonomy, it is
a restatement of the list. So the clustering here is UNSUPERVISED over surface form, the
cluster names are assigned AFTER looking at what came out, and the phrase lists used earlier
in this project are never given to the clusterer.

WHY SURFACE FORM AND NOT SEMANTICS. Register IS surface form -- the same refusal decision
rendered in different grammatical persons and stances. An embedding model would collapse
exactly the distinction we are trying to measure, because "I can't tell you how to make meth"
and "making meth is illegal" mean nearly the same thing and differ only in register. TF-IDF
over the refusal's OPENING preserves that difference; sentence embeddings would erase it.

THE GATE. If the clusters do not replicate across Zephyr and OLMo 2 -- two families sharing
no pretraining corpus, no tokenizer and no alignment recipe -- then a "register" is a
model-specific artifact, A3 is not worth running, and the whole track collapses to a
limitations paragraph. That check is the point of this script, not a footnote to it.

WHAT THE UNSUPERVISED PASS ACTUALLY FOUND, recorded because the negative is informative:
TF-IDF k-means over the openings gives silhouette ~0.03 at every k in 2..6, and the clusters
come back as sub-flavours of "I cannot" -- "cannot assist with that", "cannot fulfill your
request", "cannot provide". It does NOT recover the register split. Two reasons, both worth
stating: the pool is ~2:1 first-person so the majority register dominates the vocabulary, and
TF-IDF ranks topic words (meth, phishing, tax) alongside stance words.

The split is SYNTACTIC, not lexical. Asking who the subject of the opening clause is -- the
model itself, or the requested act -- separates the two registers perfectly where a
vocabulary method sees almost nothing. That is why `register_of` is a grammatical-person test
and not a phrase list, and it is the difference between a taxonomy and a restatement of one's
own priors.

Output: results/register_taxonomy.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from refusal_substring import (CONFUSION_SUBSTRINGS, is_refusal_strict,
                               truncate_at_turn)

# The opening is where register lives: a refusal announces its stance in the first clause and
# then explains. Using the whole completion lets topic vocabulary (meth, phishing, tax) swamp
# the stance vocabulary, and the clusters come back sorted by SUBJECT rather than by register.
OPENING_CHARS = 180


# The opening clause's grammatical subject. FIRST-PERSON means the model makes itself the
# subject ("I can't help with that", "As an AI, I cannot"); THIRD-PERSON means the requested
# act is the subject ("Bribery is illegal", "Spreading rumours is unethical", "Do not attempt
# to rob a bank"). Deliberately a closed list of opening tokens rather than a semantic test:
# it is checkable by eye, reproducible by anyone, and cannot quietly encode the conclusion.
_FIRST_PERSON = re.compile(
    r"^\W*(i|i'm|i am|i'd|i've|i'll|as an ai|as a language model|as an assistant|we|my|me)\b",
    re.I)


def register_of(text: str) -> str:
    """'first' or 'third' -- the grammatical person of the refusal's opening clause."""
    return "first" if _FIRST_PERSON.match(text.strip()) else "third"


def opening(text: str) -> str:
    t = truncate_at_turn(text).strip()
    t = re.sub(r"\s+", " ", t)
    return t[:OPENING_CHARS]


def wildguard_labels(stem: str, arm: str, n: int, comps: list[str]) -> list[bool] | None:
    """Per-item refusal labels for ONE arm, reconstructed from a judge file that stores only
    the rates and the disagreeing indices.

    The arm argument is not cosmetic. An earlier version read the 'baseline' arm for every
    call, so every ABLATED arm fell through to the substring judge -- and the substring judge
    cannot see a normative refusal, which is precisely the population the ablated arms are
    full of. The taxonomy that came back was four flavours of "I cannot", because the
    normative half had been filtered out before clustering began."""
    p = f"results/{stem}_wildguard.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if arm not in d:
        return None
    dis = set(d[arm]["disagreements"])
    sub = [is_refusal_strict(c) for c in comps]
    return [(not sub[i]) if i in dis else sub[i] for i in range(n)]


def collect() -> list[dict]:
    """Every stored completion a validated judge called a refusal, with its provenance."""
    rows = []
    for p in sorted(glob.glob("results/*_refusal.npz")
                    + glob.glob("results/*_refusal_gen*.npz")):
        z = np.load(p, allow_pickle=True)
        if "sample_completions" not in z.files:
            continue
        # keep the _gen<N> marker in the stem so the judge file matches and so
        # the two generation lengths appear as separate rows, not one.
        b = os.path.basename(p)
        m = re.match(r"^(.*?)_refusal(_gen\d+)?\.npz$", b)
        stem = m.group(1) + (m.group(2) or "")
        # Family from the lineage prefix. Was a two-way "zephyr else olmo2" guess,
        # which silently filed tulu-2 under olmo2 the moment a third family arrived --
        # and the cross-family REPLICATION claim is computed from this field.
        family = next((f for f in ("zephyr", "tulu2", "llama2", "olmo2")
                       if stem.startswith(f)), "unknown")
        sc = json.loads(str(z["sample_completions"]))
        for arm in ("baseline", "ablated"):
            comps = sc.get(arm) or []
            if not comps:
                continue
            lab = wildguard_labels(stem, arm, len(comps), comps)
            for i, c in enumerate(comps):
                # Without a judge for this arm, fall back to the substring judge -- which
                # UNDERCOUNTS the normative register by construction, so the fallback can only
                # bias the taxonomy TOWARD first-person. Recorded per row so the analysis can
                # check whether any cluster depends on fallback-labelled rows.
                refused = lab[i] if lab is not None else is_refusal_strict(c)
                # KNOWN CONTAMINANT, excluded rather than silently carried. On zephyr_base --
                # the one NON-CHAT model here -- WildGuard scores incompetence as refusal:
                # 41 of its 43 extra hits are "I'm sorry, I don't understand the question,
                # could you please rephrase it?" (O-120). Those are first-person by grammar
                # and not refusals at all, so they would inflate the first-person count of a
                # family that contributes few refusals to begin with. The substring judge's
                # confusion filter is the right instrument there, so apply it.
                if refused and any(x in c.lower() for x in CONFUSION_SUBSTRINGS):
                    refused = False
                if refused and len(opening(c)) > 30:
                    rows.append({"text": opening(c), "stem": stem, "arm": arm,
                                 "family": family, "idx": i,
                                 "judged_by": "wildguard" if lab is not None else "substring"})
    return rows


def fit(rows: list[dict], k: int, seed: int = 0):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Word n-grams, no stemming: "I cannot" vs "is illegal" is the signal, and min_df drops
    # per-prompt topic words that appear a handful of times.
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_df=0.6,
                          sublinear_tf=True, stop_words=None)
    X = vec.fit_transform([r["text"] for r in rows])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return vec, km, X


def top_terms(vec, km, k: int, n: int = 12) -> dict[int, list[str]]:
    names = np.array(vec.get_feature_names_out())
    return {c: list(names[np.argsort(km.cluster_centers_[c])[::-1][:n]]) for c in range(k)}


def register_table(rows: list[dict]) -> dict:
    """Per (checkpoint, arm): how many refusals, and the first/third split.

    The comparison that matters is BASELINE vs ABLATED on the same checkpoint. If ablation
    merely removed first-person refusals, the ablated count would fall to the baseline's
    third-person count. It does not: baselines carry ~0 third-person refusals and ablated
    arms carry ~80, so those refusals did not exist before the intervention. Ablation
    CONVERTS the register rather than filtering it."""
    agg: dict = {}
    for r in rows:
        a = agg.setdefault(r["stem"], {}).setdefault(r["arm"], {"first": 0, "third": 0})
        a[register_of(r["text"])] += 1
    return agg


def _write_report(rows: list[dict], tab: dict) -> None:
    by_fam = {}
    for r in rows:
        d = by_fam.setdefault(r["family"], {"first": 0, "third": 0})
        d[register_of(r["text"])] += 1
    out = {
        "what": "A1 -- refusal register taxonomy. Register = grammatical person of the "
                "refusal's opening clause.",
        "n_refusals": len(rows),
        "registers": ["first (model as subject: 'I can't help with that')",
                      "third (act as subject: 'Bribery is illegal')"],
        "by_family": by_fam,
        "by_checkpoint": tab,
        "unsupervised_pass": "TF-IDF k-means over openings gives silhouette ~0.03 at k=2..6 "
                             "and returns sub-flavours of 'I cannot'. It does not recover the "
                             "split: the distinction is syntactic, not lexical.",
    }
    json.dump(out, open("results/register_taxonomy.json", "w"), indent=1)
    print("wrote results/register_taxonomy.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="2,3,4,5,6")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cluster", action="store_true",
                    help="run the unsupervised pass too; it does NOT recover the split (see "
                         "the module docstring) and is kept as evidence of that")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    rows = collect()
    fam = {f: sum(r["family"] == f for r in rows) for f in ("olmo2", "zephyr")}
    fb = sum(r["judged_by"] == "substring" for r in rows)
    print(f"{len(rows)} refusals pooled | by family {fam} | "
          f"{fb} ({fb/len(rows):.0%}) labelled by the substring fallback\n")

    # --- the measurement: grammatical person, per checkpoint, baseline vs ablated
    tab = register_table(rows)
    print(f"{'checkpoint':40s} {'baseline n/1st/3rd':>24s}   {'ablated n/1st/3rd':>24s}")
    for stem in sorted(tab):
        def cell(arm):
            d = tab[stem].get(arm)
            if not d:
                return f"{'—':>24s}"
            n = d["first"] + d["third"]
            return f"{n:5d} {d['first']/n:7.0%} {d['third']/n:7.0%}" if n else f"{'—':>24s}"
        print(f"{stem:40s} {cell('baseline')}   {cell('ablated')}")
    print()

    if not args.cluster:
        if args.report:
            _write_report(rows, tab)
        return

    from sklearn.metrics import silhouette_score
    for k in (int(x) for x in args.k.split(",")):
        vec, km, X = fit(rows, k)
        sil = silhouette_score(X, km.labels_, sample_size=min(2000, X.shape[0]),
                               random_state=0)
        sizes = np.bincount(km.labels_, minlength=k)
        print(f"k={k}  silhouette {sil:+.3f}  sizes {list(sizes)}")
        for c, terms in top_terms(vec, km, k, 8).items():
            # Per-cluster family mix: a cluster that is ~100% one family is a model quirk,
            # not a register. This is the stability gate in its cheapest form.
            mix = {f: sum(1 for r, l in zip(rows, km.labels_) if l == c and r["family"] == f)
                   for f in ("olmo2", "zephyr")}
            tot = sum(mix.values())
            print(f"    c{c} n={tot:4d} olmo2 {mix['olmo2']/tot:5.0%} zephyr "
                  f"{mix['zephyr']/tot:5.0%} | {', '.join(terms)}")
        print()

    if args.report:
        _write_report(rows, register_table(rows))
        k = int(args.k.split(",")[0])
        vec, km, X = fit(rows, k)
        out = {"n": len(rows), "k": k, "family_counts": fam,
               "opening_chars": OPENING_CHARS,
               "fallback_labelled": fb,
               "clusters": {}}
        for c in range(k):
            members = [r for r, l in zip(rows, km.labels_) if l == c]
            mix = {f: sum(1 for r in members if r["family"] == f) for f in ("olmo2", "zephyr")}
            out["clusters"][str(c)] = {
                "n": len(members), "family_mix": mix,
                "top_terms": top_terms(vec, km, k, 15)[c],
                "examples": [m["text"][:160] for m in members[:: max(1, len(members) // 8)]][:8],
            }
        json.dump(out, open("results/register_taxonomy.json", "w"), indent=1)
        print("wrote results/register_taxonomy.json")


if __name__ == "__main__":
    main()
