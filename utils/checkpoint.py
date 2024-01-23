import heapq
import os
import torch

class Checkpointer:
    def __init__(self, path, top_k=5, keep_min=True):
        super(Checkpointer, self).__init__()

        self.heap = [] # heap of (loss/acc, ckpt_file)
        self.keep_min = keep_min # True for loss, False for acc
        self.top_k = top_k # select top-k ckpt_files
        self.ckpt_path = os.path.join(path, "checkpoints")
        os.makedirs(self.ckpt_path, exist_ok=True)

    def load(self, model, ckpt_file=None):
        if ckpt_file:
            ckpt_file = os.path.join(self.ckpt_path, ckpt_file)
            model.load_state_dict(torch.load(ckpt_file)["model_state_dict"])
        return model

    def save(self, model, ckpt_file, criterion):
        ckpt_file = os.path.join(self.ckpt_path, ckpt_file)
        torch.save({"model_state_dict": model.state_dict()}, ckpt_file)
        
        # Maintain top-k checkpoints
        if self.keep_min:
            criterion = -criterion # reverse for max-heap
        heapq.heappush(self.heap, (criterion, ckpt_file))
        if len(self.heap) > self.top_k:
            _, ckpt_file = heapq.heappop(self.heap)
            os.remove(ckpt_file)

    def best_ckpt_file(self):
        if not self.heap:
            raise ValueError("No records found (only call it while training).")
        return max(self.heap)[1]