from unsloth import FastLanguageModel

MODEL_PATH = "adapters/stage3_dpo"  # relative to the repo root when cloned

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=2048,
    dtype=None, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


def generate_answer(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=200, temperature=0.7)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    question = "How can I apply for a refund?"
    print(generate_answer(question))
