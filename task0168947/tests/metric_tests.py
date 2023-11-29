import torch
import numpy as np

from sklearn.metrics import f1_score
from torchmetrics.classification import MulticlassF1Score

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("Simple Test")
N_classes = 3
average = 'macro'

true = [0] * 5
preds = [0] * 5

true[0] = 1
preds[0] = 1

metric = MulticlassF1Score(num_classes=N_classes, average=average).to(device)
val = metric(torch.tensor(preds, device=device), torch.tensor(true, device=device))
print(f"Torch F1: {val:0.6f}")

val = f1_score(true, preds, average=average)
print(f"Sklearn F1: {val:0.6f}")

metric = MulticlassF1Score(num_classes=N_classes, average=None).to(device)
val = metric(torch.tensor(preds, device=device), torch.tensor(true, device=device))
print("Torch F1: val")


print("More complicated test")

N_classes = 100
average = 'macro'

true = [0] * 100
preds = [0] * 100

true_idxs = [17, 42,56 ,32, 41, 77, 67, 54, 81]
for i in true_idxs:
    true[i] = i

pred_idxs = [11, 2, 6, 16, 45, 32, 67, 54, 88, 97, 53, 21, 84]
for i in pred_idxs:
    preds[i] = i

metric = MulticlassF1Score(num_classes=N_classes, average=average).to(device)
val = metric(torch.tensor(preds, device=device), torch.tensor(true, device=device))
print(f"Torch F1: {val:0.6f}")

val = f1_score(true, preds, average=average)
print(f"Sklearn F1: {val:0.6f}")

metric = MulticlassF1Score(num_classes=N_classes, average=None).to(device)
val = metric(torch.tensor(preds, device=device), torch.tensor(true, device=device))
print("Torch F1: val")


