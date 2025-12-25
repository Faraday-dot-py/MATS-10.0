# latent_llama_chain.py
#
# Idea:
#   - Use a real LLaMA model on CPU.
#   - Encode the user's prompt ONCE to get an initial hidden state x_0.
#   - For step k = 1..N:
#         * Run ONE full pass: x_k, logits_k = Model(x_{k-1})
#           (using inputs_embeds = x_{k-1}, NOT tokens)
#         * From logits_k, read out ONE token (for us to print later).
#         * Feed x_k (hidden state) back in next step.
#   - We NEVER feed generated text/tokens back into the model.
#   - At the end, we decode the N generated token IDs and show:
#         prompt + generated_text
#
# Requirements (run once, outside this file):
#   pip install "transformers>=4.43.0" torch accelerate
#
# You also need access to a small LLaMA model, e.g.:
#   meta-llama/Llama-3.2-1B-Instruct
# (accept the license on Hugging Face, then `huggingface-cli login`).

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cpu"

DEFAULT_LATENT_PASSES = 20   # how many hidden-state steps = how many tokens
DEFAULT_TEMPERATURE = 0.8


def sample_next_token(logits_row: torch.Tensor, temperature: float = 1.0) -> int:
    """Sample a token ID from a [vocab_size] logits vector."""
    if temperature <= 0:
        return int(torch.argmax(logits_row).item())
    probs = F.softmax(logits_row / temperature, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return int(token_id.item())


def main():
    print("Loading tokenizer and model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # CPU-safe
    )
    model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 1. User prompt + settings (no CLI args) ----
    prompt = "Hello! "
    latent_passes = DEFAULT_LATENT_PASSES
    temperature = DEFAULT_TEMPERATURE

    # ---- 2. Encode prompt ONCE ----
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(DEVICE)         # [1, T]
    attention_mask = encoded["attention_mask"].to(DEVICE)  # [1, T]

    # ---- 3. Get initial hidden state x_0 via embeddings ----
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()      # nn.Embedding
        x = embed_layer(input_ids)                      # [1, T, hidden_dim]

    print(f"Initial hidden state shape: {x.shape}")

    generated_ids = []  # we'll store the N generated token IDs here

    # ---- 4. Latent chain: one full pass per hidden input ----
    with torch.no_grad():
        for step in range(latent_passes):
            outputs = model(
                inputs_embeds=x,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # New hidden state for next step
            x = outputs.hidden_states[-1]               # [1, T, hidden_dim]

            # Readout: logits for this step, last position
            logits = outputs.logits                     # [1, T, vocab_size]
            last_logits = logits[0, -1, :]              # [vocab_size]
            
            next_token_id = sample_next_token(last_logits, temperature)
            generated_ids.append(next_token_id)

            # NOTE: we DO NOT feed this token back in. Only x (latent) recurs.

            # Optional debug:
            # print(f"Step {step+1}: token id {next_token_id}, norm(x)={x.norm().item():.3f}")

    # ---- 5. Decode ONCE at the end ----
    generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.long)
    generated_text = tokenizer.decode(
        generated_ids_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print("\n--- Result ---")
    print("Prompt:")
    print(prompt)
    print("\nGenerated (latent chain, no text fed back):")
    print(generated_text)
    print("\nPrompt + generated:")
    print(prompt + generated_text)
    for id in generated_ids:
        print(f"{id}: {tokenizer.decode(id)}")
    print(generated_ids)    


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
