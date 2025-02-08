import torch

def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        print(f"Prompt shape: {prompt.shape}")
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
        print(f"Output hidden states: {output.hidden_states}")  # Debug hidden states

    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
    hidden_states = hidden_states.detach().cpu().numpy()

    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            #print(f"Collector states shape: {states_per_gen.shape}")  # Debug collector output
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()

    mlp_wise_hidden_states = []
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

