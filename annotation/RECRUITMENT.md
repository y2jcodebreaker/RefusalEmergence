# What to tell the annotators

Three messages, in order. Keep the first one short — it is a yes/no decision, not a briefing.

The one thing not to say, at any stage: **what we expect the answers to be.** Not "we think the
automatic judge misses refusals", not "identity refusals should survive". They are told the
purpose in the codebook, in the form that cannot bias them. Everything below is written to that
constraint.

---

## 1 · The ask

> Hi — I'm looking for two people to do a labelling pass for a paper I'm submitting to ICML in
> January, and I'd rather ask directly than post it.
>
> It's about 90 minutes of reading, in one or two sittings, whenever suits you this month. You
> read a request someone sent to a language model and what the model wrote back, and answer two
> questions: did it refuse, and if so, in what style. There are ten keyboard keys and no setup —
> one file you double-click.
>
> Two things to know before you say yes. **The requests are genuinely nasty** — fraud, weapons,
> self-harm, hate speech — and some of the model answers go along with them. You'd be reading
> that material. Saying no to this is a completely reasonable answer and costs me nothing.
> **And I need two people who aren't me**, because the whole point is that the categories hold
> up for someone who didn't invent them. If you do it, you'd be named in the acknowledgements,
> or as a co-author if you'd rather be involved further.
>
> Here's a ten-item demo with harmless stand-in items, so you can see exactly what the task
> feels like before deciding: https://claude.ai/artifact/M8eGK23qe2FhzjsrXtp1ua
>
> No rush, and no hard feelings either way.

Send the demo link with the ask, not after it. It answers "what am I actually agreeing to"
better than another paragraph will.

---

## 2 · When they say yes

Attach `CODEBOOK.md`, `calibration.html`, and their own `sheet_X.html`. One file each —
annotator A gets `sheet_A.html` and never `sheet_B.html`.

> Thank you. Three files:
>
> 1. **`CODEBOOK.md`** — read this first, it takes about ten minutes. It defines the two
>    questions and rules on the cases that are genuinely ambiguous, so you aren't left guessing.
> 2. **`calibration.html`** — 20 items with my answers hidden behind a click. Decide yours
>    first, then open each one. These are deliberately the hard cases; the real sheet is easier.
>    **If you disagree with several of them, tell me before you start the real sheet.** That
>    means my codebook is unclear, which is mine to fix and not yours to work around.
> 3. **`sheet_X.html`** — the real one. Double-click it. 340 items, about 90 minutes. No
>    internet needed and nothing is uploaded. Your progress saves after every item, so close
>    the tab whenever you want. At the end click **Download labels** and send me the file.
>
> Two things on saving, because I'd rather over-explain than lose your afternoon. **If a red
> banner says the browser isn't saving, believe it** — some browsers refuse storage for a file
> opened from disk. Press **Export** (or `ctrl+E`) now and then and keep those files, or reopen
> the sheet in Chrome or Firefox. And **you can stop whenever** — press Export and send what you
> have. A partial sheet is useful; please don't grind to the end on my account.
>
> Four rules, and they all matter more than speed:
>
> - **Don't discuss any item with the other annotator, or with me, until you've both sent your
>   files back.** Where the two of you disagree is the actual measurement. A conversation
>   beforehand erases it.
> - **Don't look anything up.** I want your reading, not a researched answer.
> - **Take breaks.** The sheet records how long each item took, and I'd rather see two sittings
>   than one rushed hour. That timing is for reporting the labelling conditions honestly in the
>   paper, not for checking up on you.
> - **If an item is genuinely 50/50, pick the one you'd defend and move on.** Don't agonise.
>   The items you two split on get looked at separately anyway.
>
> If something is unclear in the codebook, ask me — about the *definitions*, any time. Just not
> about specific items.

---

## 3 · After both files are back

Only then. `annotate_score.py --phase kappa` runs before any of this conversation happens,
because the agreement number has to be fixed before anyone argues about individual items.

> Both files are in and the agreement number is computed and saved, so now we can talk about
> the ones you two saw differently. There were N of them. Here they are — I'd like to hear your
> reasoning rather than have you converge on mine.

Tell them the κ afterwards. They have earned knowing how it came out, including if it came out
badly — in which case the honest result is that the categories don't transfer, and that goes in
the paper as a finding rather than being quietly re-run with a rewritten codebook.

---

## Who may not annotate

- **Any author.** The risk `provenance.py` names for this control is *"the register taxonomy
  being the authors' reading"*, so an author labelling it tests nothing.
- **`claude-opus-5`.** Already the unblinded labeller of record in
  `results/b1_v2_residual_audit.json`, which is why that file is the calibration pack and not
  the control.
- Anyone who has seen the register table, the draft §3, or the machine labels.

Domain expertise is explicitly **not** required. With the codebook in hand this is a careful
reading task, which is the point — a taxonomy that only works for people who already know the
hypothesis is not a taxonomy.
