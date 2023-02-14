import torch


def compare(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

model_31 = torch.load("ep-31.pt")
model_32 = torch.load("ep-32.pt")
print(model_31)
print(f"model31 and 32: {compare(model_31,model_31)}")

# print(model_31)
# print(model_32)
