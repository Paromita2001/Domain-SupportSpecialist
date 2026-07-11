# Fine-Tuning Concepts — In My Own Words

## Why full fine-tuning is expensive
Full fine-tuning updates every weight in the model, which means the optimizer has to
keep a copy of the gradients and momentum/variance state for every one of those
weights too — for a 1.5B parameter model that's several times the model's own size
sitting in GPU memory at once, on top of the activations from the forward pass. That
memory footprint is why full fine-tuning of even a "small" LLM needs a datacenter-class
GPU, not the free T4 this project ran on.

## What LoRA does
LoRA (Low-Rank Adaptation) freezes all of the original model weights and instead
injects a small pair of low-rank matrices next to each target weight matrix (here,
the attention and MLP projections). Only those small matrices are trained. Because
they're low-rank, the number of trainable parameters drops by orders of magnitude
compared to full fine-tuning, while the frozen base model still does most of the
heavy lifting.

## What QLoRA adds
QLoRA takes LoRA a step further by loading the frozen base model in 4-bit precision
instead of 16/32-bit, then training the LoRA adapters (still in higher precision) on
top of that quantized base. The frozen weights barely lose usable quality at 4-bit,
but they take a quarter of the memory to store.

## Why that matters on a free Colab T4
A T4 has 16GB of VRAM. Loading a 1.5B model at full precision plus optimizer state
for full fine-tuning would not fit. Loading the same model 4-bit-quantized (QLoRA)
plus small LoRA adapters comfortably fits with room left for a reasonable batch size
— which is the only reason this whole project was possible on a free-tier GPU instead
of a paid cloud instance.

## What is non-instruction fine-tuning?
Training the model on plain domain text with no instruction/response structure at
all — just paragraphs of the kind of language the domain uses. The goal isn't to
teach the model to follow commands, it's to shift its internal sense of vocabulary,
tone, and typical phrasing toward the support domain before it ever sees a labeled
question-answer pair.

## What is instruction fine-tuning?
Training on paired (instruction, response) examples, formatted through the model's
chat template, so the model learns the actual task: given a user's question, produce
a direct, correctly-formatted, on-topic answer. This is what turns a general-purpose
chat model into something that behaves like a support assistant specifically.

## What is DPO?
DPO (Direct Preference Optimization) trains directly on pairs of (chosen, rejected)
responses to the same prompt, nudging the model's output distribution toward the
chosen response and away from the rejected one — without needing a separate reward
model the way classic RLHF does. It optimizes relative preference, not an absolute
"correct answer" target.

## Difference between SFT and DPO
SFT teaches the model "here is a correct response to imitate" for a given input —
it only ever sees one target per example. DPO teaches the model "this response is
better than that one for the same input" — it sees both a good and a bad answer
together, which is a more direct way to fix over-generic, unhelpful, or unsafe
behavior that SFT alone doesn't always remove.

## Hyperparameters I actually used
| Setting | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Learning rate (Stage 1, non-instruction) | 2e-4 |
| Learning rate (Stage 2, SFT) | 2e-4 |
| Learning rate (Stage 3, DPO) | 5e-6 |
| Batch size (effective) | 8 (2 per device x 4 gradient-accumulation steps) |
| DPO beta | 0.1 |
