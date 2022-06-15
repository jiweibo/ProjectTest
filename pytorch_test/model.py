import torch
import torchvision

model = torchvision.models.resnet18()

# output fp16 model
model.half()

example = torch.rand((1, 3, 224, 224), dtype=torch.float16)

traced_script_module = torch.jit.trace(model, example)
print(traced_script_module)
#output = traced_script_module(torch.ones(1,3,224,224))
#print(output[0, 0:5])
traced_script_module.save("traced_resnet_model.pt")
