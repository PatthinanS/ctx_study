from transformers import AutoModel
import torch

a = AutoModel.from_pretrained("roberta-base")
b = AutoModel.from_pretrained("roberta-base")
# identical across loads → weights come from checkpoint (saved, but possibly still untrained)
# different across loads → newly random-initialized each time
print(torch.equal(a.pooler.dense.weight, b.pooler.dense.weight))