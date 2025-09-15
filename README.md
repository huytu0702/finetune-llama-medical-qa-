# Vietnamese Medical QA Finetuning

Fine‑tune **Llama 3.2 Instruct** on the `hungnm/vietnamese-medical-qa` dataset using **Unsloth (LoRA, 4‑bit)** and **TRL SFTTrainer**.

---

## 🚀 Quick Start

```bash
pip install unsloth bitsandbytes accelerate xformers trl peft transformers datasets sentencepiece
```

---

## ⚙️ Setup

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)
```

---

## 📚 Dataset

Hugging Face: `hungnm/vietnamese-medical-qa`

```python
from datasets import load_dataset

template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"

def format(ex):
    return {"text": [template.format(q=q,a=a) for q,a in zip(ex["question"],ex["answer"])]}

data = load_dataset("hungnm/vietnamese-medical-qa", split="train").map(format, batched=True)
```

---

## 🏋️ Train

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data,
    dataset_text_field="text",
    args=SFTConfig(output_dir="outputs", num_train_epochs=1),
)
trainer.train()
```

---

## 🔎 Inference

```python
FastLanguageModel.for_inference(model)

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHỏi bác sĩ...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
outputs = model.generate(**tokenizer([prompt], return_tensors="pt").to("cuda"), max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

---

## ⚠️ Disclaimer

For **research/demo** only. Not for real medical use.
