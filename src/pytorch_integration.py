import torch
import taichi as ti

torch.random.manual_seed(1)
ti.init(ti.cuda, default_fp=ti.f32)
data_field = ti.field(ti.f32, shape=2)
weight_field = ti.field(ti.f32, shape=2, needs_grad=True)
output_field = ti.field(ti.f32, shape=2, needs_grad=True)
loss = ti.field(ti.f32, (), needs_grad=True)

data_tensor = torch.rand(2, dtype=torch.float32).cuda()
weight_tensor = torch.ones(2, dtype=torch.float32).cuda()
data_field.from_torch(data_tensor)
weight_field.from_torch(weight_tensor)


@ti.kernel
def multiply():
    for i in output_field:
        output_field[i] = weight_field[i] * data_field[i]


@ti.kernel
def calc_loss():
    for i in output_field:
        loss[None] += output_field[i] ** 2


@ti.kernel
def clear_grads():
    loss.grad[None] = 0.0
    for i in output_field:
        output_field.grad[i] = 0.0
        weight_field.grad[i] = 0.0


print(f"data = {data_tensor}")
print(f"weights = {weight_tensor}")
multiply()
calc_loss()
loss.grad[None] = 1.0
calc_loss.grad()
multiply.grad()
print(f"weight field grad calc by kernel.grad: {weight_field.grad}")

with ti.Tape(loss):
    multiply()
    calc_loss()

print(f"weight field grad calc by ti.Tape: {weight_field.grad}")

clear_grads()
cuda = torch.device("cuda")
output_tensor = output_field.to_torch(device=cuda).requires_grad_(True)
torch_loss = (output_tensor ** 2).sum()
torch_loss.backward()
output_field.grad.from_torch(output_tensor.grad)
multiply.grad()
print(f"weight field grads calc by hybrid pipeline: {weight_field.grad}")
