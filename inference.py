import torch
from transformers import GPT2TokenizerFast
from train import TransformerWithMoE
from fmoe.transformer import FMoETransformerMLP
from fmoe.gates import GShardGate

def generate_text(model, tokenizer, prompt, max_new_tokens=50, device="cuda", temperature=1.0, top_k=50):
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_trim = generated[:, -1024:]  # не больше позиции pos_emb
            outputs = model(input_trim)
            logits = outputs[:, -1, :] / temperature

            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, indices, values)
                probs = torch.softmax(probs, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def load_model(vocab_size, checkpoint_path, device="cuda"):
    d_model = 256
    num_layers = 3
    top_k = 2
    num_experts_per_device = 5
    world_size = 2
    total_experts = num_experts_per_device * world_size

    model = TransformerWithMoE(vocab_size, d_model, num_layers, total_experts, top_k)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    vocab_size = tokenizer.vocab_size + 1  # pad_token добавлен

    checkpoint_path = "checkpoint_epoch_5.pt"  # путь к сохранённому чекпоинту
    model = load_model(vocab_size, checkpoint_path, device=device)

    prompt = "My name is"
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        device=device,
        temperature=2.,
        top_k=5
    )

    print("\n📝 Generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()
