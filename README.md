# Domain-SupportSpecialist : A Domain-Tuned Customer Support Assistant
### Three-stage LLM fine-tuning with Unsloth: Non-Instruction FT → SFT → DPO

## 1. Domain selected
**Customer Support Assistant** — answers customer questions about refunds, order
tracking, cancellations, replacements, delivery, and payment issues.

## 2. Business problem
Support tickets overwhelmingly repeat the same ~15 request types: cancel an order,
track a refund, report a duplicate charge, ask about a delivery window, request a
replacement instead of a refund. A generic AI assistant handles these vaguely —
it deflects to "contact customer service" instead of actually explaining the
process, because it has no knowledge of the company's specific policies, tone, or
terminology. That's expensive: every one of those generic non-answers still ends
up needing a human agent anyway.

## 3. Proposed solution
Fine-tune a small open-source LLM (Qwen2.5-1.5B-Instruct) in three progressive
stages so it moves from "generic chatbot" to "specialized support agent":

1. **Non-instruction fine-tuning** on raw support-response text, so the model
   absorbs domain vocabulary and tone before it ever tries to answer anything.
2. **Instruction fine-tuning (SFT)** on real (question, answer) pairs, so it
   learns the actual task — directly answering a support question instead of
   deflecting.
3. **DPO preference alignment** on (question, good answer, bad answer) triples,
   so it learns to prefer specific, helpful responses over vague or evasive ones.

Each stage is trained with LoRA/QLoRA so the whole pipeline runs on a **free
Google Colab T4 GPU** — no paid infrastructure required. The result is compared
against the untouched base model on a fixed set of 10 test questions, at every
stage, so the improvement (or lack of it) is measured, not assumed.

## 4. Dataset details
Derived from the public [Bitext customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(Hugging Face Hub, **license: CDLA-Sharing-1.0**, verified directly from the
dataset's metadata). The full dataset has ~26,872 rows across 27 intents; we
filtered to the **14 intents relevant to this domain** (cancel/change order,
refund policy/status, payment methods/issues, delivery options/period, order
tracking, shipping address changes, complaints, human-agent escalation) — those
14 intents total 13,896 rows, the real pool this project samples from.

Cleaning performed before use: explored the raw data first (shape, dtypes,
missing values, duplicates, the dataset's own `flags` tagging scheme),
resolved placeholder tags like `{{Order Number}}` and `{{Customer Support
Phone Number}}` into realistic values, excluded rows flagged `W` (offensive
language) for safety, dropped responses under 40 characters, and manually
verified a random sample of the cleaned output before trusting it. The full,
runnable version of this pipeline (download → EDA → clean → split) is at
`data/data_preparation.ipynb`.

| File | Size | Shape | Used by |
|---|---|---|---|
| `data/non_instruction_data.txt` | 184 paragraphs | plain text, no Q/A | Stage 1 |
| `data/instruction_dataset.jsonl` | 200 pairs | `{instruction, response}` | Stage 2 |
| `data/preference_dataset.jsonl` | 100 triples | `{prompt, chosen, rejected}` | Stage 3 |

All three files are **disjoint** — no response text repeats across them, so no
stage is just re-learning content another stage already memorized.

## 5. Base model
`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` — a 1.5-billion-parameter
general-purpose chat model, loaded in 4-bit throughout (QLoRA).

## 6. Non-instruction fine-tuning approach
`notebooks/non_instruction_finetuning.ipynb`. Loads the base model, chunks
`non_instruction_data.txt` into 184 individual paragraph-level training
examples, attaches a LoRA adapter, and trains for a short, deliberately capped
60 steps (a "warm-up," not the graded core) — 184 examples over 3 epochs at
that step cap. Goal: shift the model's internal vocabulary and tone toward the
support domain before it ever sees a question-answer pair. Adapter saved to
`adapters/stage1_non_instruction/`.

Actual per-step training loss from this run:

| Step | Training Loss |
|---|---|
| 5 | 1.939523 |
| 10 | 2.025378 |
| 15 | 1.705911 |
| 20 | 1.527913 |
| 25 | 1.488091 |
| 30 | 1.312090 |
| 35 | 1.178858 |
| 40 | 1.151445 |
| 45 | 1.196575 |
| 50 | 0.952569 |
| 55 | 1.014675 |
| 60 | 1.026075 |

Loss trends down overall (1.94 → ~1.0) with a bit of noise along the way
(a small bump at step 10 and again at step 45) — expected for a short,
184-example warm-up run rather than a smooth curve.

## 7. Instruction fine-tuning approach
`notebooks/instruction_finetuning.ipynb`. Loads the Stage-1 adapter as its
starting point, formats all 200 instruction/response pairs through the model's
chat template, and trains for 3 full epochs (75 steps). This is the stage that
actually teaches the model to answer a support question directly instead of
just sounding right. Adapter saved to `adapters/stage2_sft/`.

Actual per-step training loss from this run:

| Step | Training Loss |
|---|---|
| 10 | 1.576857 |
| 20 | 0.987686 |
| 30 | 0.887399 |
| 40 | 0.795725 |
| 50 | 0.741381 |
| 60 | 0.674428 |
| 70 | 0.648318 |

A clean, steady drop from 1.58 → 0.65 across 70 steps — smoother than
Stage 1's, as expected once the model is learning an actual structured task
(question → answer) instead of just absorbing tone from raw paragraphs.

## 8. DPO alignment approach
`notebooks/dpo_alignment.ipynb`. Loads the Stage-2 SFT adapter, attaches a
fresh LoRA specifically for this stage, and trains on all 100 preference
triples for 2 epochs (26 steps) using `DPOTrainer` (`beta=0.1`, `ref_model=None`
— reuses the same model with adapters disabled as the reference, avoiding a
second full model copy in memory). Instead of imitating one correct answer,
the model sees a chosen-vs-rejected pair for the same question and is nudged
toward the chosen one — catching vague or evasive answers that SFT alone
didn't fully remove. Adapter saved to `adapters/stage3_dpo/` — this is the
**final, deployed model**.

Actual per-step training log from this run:

| Step | Loss | rewards/chosen | rewards/rejected | rewards/accuracies | rewards/margins |
|---|---|---|---|---|---|
| 5 | 0.027122 | 7.976673 | -0.346258 | 1.000000 | 8.322931 |
| 10 | 0.016365 | 7.120435 | -0.476984 | 1.000000 | 7.597420 |
| 15 | 0.030957 | 8.781938 | -0.560052 | 1.000000 | 9.341990 |
| 20 | 0.024412 | 7.617857 | -0.617353 | 1.000000 | 8.235210 |
| 25 | 0.008485 | 8.730375 | -0.554893 | 1.000000 | 9.285269 |

What these columns actually mean: `rewards/chosen` is how strongly the model
now favors the "good" answer, `rewards/rejected` is how strongly it favors
the "bad" one (correctly negative throughout — it's being pushed away from
those). `rewards/accuracies = 1.000000` on every logged step means the model
ranked the chosen answer above the rejected one on 100% of training examples
in that batch. `rewards/margins` (the gap between chosen and rejected) stays
large and positive throughout (~7.6–9.3), and training loss is already very
low and near-zero variance by step 25 — DPO converged quickly on this
100-example preference set, which is expected for a small, clean dataset
(see Section 12 for what this does and doesn't prove about real-world
answer quality).

## 9. LoRA / QLoRA configuration
| Setting | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Base weight precision | 4-bit (QLoRA) |
| Trainable parameters | 18,464,768 / 1,562,179,072 (**1.18%** of the model) |
| Learning rate (Stages 1–2) | 2e-4 |
| Learning rate (Stage 3, DPO) | 5e-6 |
| Effective batch size | 8 (2 per device × 4 gradient-accumulation steps) |
| DPO beta | 0.1 |

Full concept explanation (why LoRA, why QLoRA, SFT vs DPO) in
`reports/fine_tuning_explanation.md`.

## 10. Training logs
Confirmed from actual runs on a Colab Tesla T4:

```
Stage 1 — Num examples = 184 | Num Epochs = 3 | Total steps = 60 | train_loss = 1.38
Stage 2 — Num examples = 200 | Num Epochs = 3 | Total steps = 75
Stage 3 — Num examples = 100 | Num Epochs = 2 | Total steps = 26
Trainable parameters = 18,464,768 of 1,562,179,072 (1.18% trained)  [all stages]
```
`[Training screenshot placeholder — paste a screenshot of any notebook's
training-loss cell output here before final submission, e.g.
![training log](reports/training_log.png)]`

## 11. Before vs after output comparison
Full 10-question, three-way comparison — with every judgment column filled in
by hand — in `reports/base_model_evaluation.md`, `reports/sft_model_comparison.md`,
and `reports/final_evaluation.md`.

**Final tally across the 10 fixed test questions:**
- Base vs SFT: **SFT wins 8/10**, Base wins 2/10 — not ties, genuine Base
  wins on two specific rows (see below).
- Base vs SFT vs DPO: **DPO wins 7/10**, SFT wins 2/10, Base wins 1/10 — not a
  perfect sweep, by design (see below).

**Q: "How long does a refund take to appear on my card?"**
- *Base:* punts entirely — refuses to engage and redirects to the customer's
  own bank.
- *DPO:* gives a concrete number (7–14 business days for card refunds) — the
  only one of the three with an actual figure attached.

**Q: "My package says delivered but I never received it, what do I do?"**
- *Base* wins this row in both comparisons — it offers concrete
  troubleshooting steps (check delivery status, contact the shipping
  carrier), while both SFT and DPO only ask for a tracking number without
  giving any guidance first. An honest example of fine-tuning *not*
  improving every case.

**Q: "I was charged twice for one order, how do I fix this?"**
- *SFT* actually gives a **misleading** fix here — it tells the customer to
  click "Edit Order," which wouldn't undo a duplicate charge at all.
  *DPO* catches this and instead points to a phone/Live Chat contact to
  investigate — less procedurally detailed, but the more accurate answer.
  A genuinely useful example of DPO correcting a real SFT mistake, not just
  making answers sound nicer.

**Q: "Can I get a replacement instead of a refund?"**
- *Base* is flatly wrong here — it states replacements "are generally not
  available," which contradicts real return-policy norms. *SFT* wins this
  row by explicitly confirming ("Absolutely!") that replacements are
  possible; *DPO*'s answer is friendlier in tone but never actually
  confirms it either way.

## 12. Final observations
- Fine-tuning visibly shifted the model from generic deflection ("contact
  customer service", "I am not able to provide real-time information") toward
  specific, step-by-step, company-toned answers (referencing "Order History,"
  concrete timeframes, a support phone number) by Stage 2, with DPO further
  sharpening directness on most — not all — questions: SFT beat Base on
  8/10 questions, and DPO was judged the best of all three on 7/10 questions.
- The base model didn't just sound generic in a few places — it stated a
  flatly **wrong** policy twice ("replacements are generally not available,"
  a duplicate-charge instruction that wouldn't actually fix anything once
  SFT introduced its own "Edit Order" mistake). DPO caught and corrected the
  SFT mistake specifically, which is a more concrete example of DPO's value
  than just "sounds nicer."
- The one row where fine-tuning consistently lost (across both comparisons)
  was a case where both SFT and DPO defaulted to asking a clarifying
  question ("could you provide the tracking number?") instead of giving any
  immediate guidance, while the untrained base model happened to offer a
  generic-but-actionable checklist. A real, reproducible failure mode worth
  knowing about rather than hiding.
- DPO's training reward accuracy converged quickly on the 100-example
  preference set, which is expected for a small, clean preference dataset —
  but it's a training-data metric, not a guarantee every real answer improved,
  which is exactly why the 10-question manual review still mattered.
- Unsloth's `"Already have LoRA adapters! We shall skip this step"` message
  during Stage 2 is not an error — it means Stage 2 continued training
  Stage 1's adapter directly rather than creating a redundant second one.

## 13. Challenges faced
- **Library version conflict:** an early install cell pinned `trl<0.9.0` to
  avoid one issue, but that pin directly conflicted with `unsloth_zoo`'s
  actual requirements — causing a `PicklingError` mid-training and later a
  `Trainer.__init__() got an unexpected keyword argument 'tokenizer'` error.
  Fixed by removing the old pin and switching to the current
  `processing_class=tokenizer` argument modern `trl` expects.
- **Google Drive folder placement:** the project folder was accidentally
  uploaded one level too deep in Drive during setup, which caused confusing
  `FileNotFoundError`s several cells into a notebook rather than immediately
  — fixed by adding an explicit folder-structure check as the very first cell
  of every notebook.
- **Free-tier GPU quota:** hit Colab's free daily GPU quota mid-project,
  requiring a switch to a different Google account to keep training moving
  (only needing to re-upload the small prerequisite files for whichever
  stage was still pending, not the whole project, kept this fast).
- **A notebook auto-copy bug that silently overwrote the wrong file:** each
  notebook originally had a convenience cell that copied "whichever `.ipynb`
  was most recently modified in Drive" back into `notebooks/` after a run.
  Once more than one notebook tab was open in the same session, that logic
  occasionally grabbed the wrong file, overwriting a real training notebook
  with unrelated test code. Fixed by removing that automation entirely and
  manually downloading each executed notebook directly from its own Colab
  tab instead — a good reminder that "most recently modified" is not the
  same as "the one I actually mean."

## 14. Future improvements
- Scale the preference dataset well beyond 100 pairs and add a wider range of
  rejected-answer failure modes (unsafe, rude, factually wrong) rather than
  mostly generic-vs-specific contrasts.
- Try ORPO as an alternative to DPO and compare convergence/quality directly.
- Expand evaluation beyond the fixed 10 questions with automated scoring
  (e.g. an LLM-as-judge pass) to catch regressions a small manual set might miss.
- Add a lightweight web demo (Gradio/Streamlit) on top of `src/inference.py`
  for live use instead of a script-only interface.

## Running this project
1. Upload/clone this repo into Google Drive.
2. Open `notebooks/non_instruction_finetuning.ipynb` in Colab, `Runtime -> Change
   runtime type -> T4 GPU`, run all cells top to bottom.
3. Open `notebooks/instruction_finetuning.ipynb`, run all cells.
4. Open `notebooks/dpo_alignment.ipynb`, run all cells.
5. The three `reports/*.md` comparison tables are pre-filled with real judgments
   from this project's own run — re-run and re-judge them yourself if you
   retrain on different data.

### Running in Colab after mounting Drive
- Upload this whole `Domain-SupportSpecialist/` folder into your Google Drive
  (drag-and-drop at drive.google.com, or `git clone` it directly into Drive from a
  Colab cell).
- In any notebook: `from google.colab import drive; drive.mount('/content/drive')`,
  then set `PROJECT = '/content/drive/MyDrive/Domain-SupportSpecialist'`
  (adjust the path if uploaded under a different folder name).
- Open the `.ipynb` files directly from Drive with "Open with -> Google Colaboratory" —
  they already assume `PROJECT` points at this folder, so no path edits are needed if
  the folder name matches.
