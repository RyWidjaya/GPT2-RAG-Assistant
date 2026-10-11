import torch 

@torch.no_grad()
# pass in model, enc, device, eos_id, max_seq_len so that in run_demo we can pass these through 
# the previous implementation in jupyter did not need to pass these as arguments as thye were already initialised previous notebook cells
def ask(model, enc, device, eos_id, max_seq_len, context, question, max_new_tokens=48):
    model.eval()
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    ids = enc.encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -max_seq_len:])
        next_id = int(logits[0, -1].argmax())
        if next_id == eos_id:
            break
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)
    return enc.decode(idx[0, len(ids):].tolist()).strip()