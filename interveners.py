import torch
import torch.nn as nn

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        print(f"Received args: {args}, kwargs: {kwargs}")  # Debugging the input
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False 
     
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
        
    def reset(self):
        self.states = []
        self.actions = []
        
    def __call__(self, b, s):
        if self.head == -1:
            # Collect all heads, reshape to (num_heads, head_dim)
            # If we have 4096 features and 32 heads, we expect each head to have 128 dimensions
            self.states.append(b[0, -1].reshape(32, 128).detach().clone())  # Reshape to (num_heads, head_dim)
        else:
            # Collect only the specific head's activations
            self.states.append(b[0, -1].reshape(32, 128)[:, self.head].detach().clone())  # Reshape and select the specific head
        return b

    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b