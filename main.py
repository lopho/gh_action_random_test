import torch
import os

gen = torch.Generator()
gen.manual_seed(0)
ts = []
for i in range(1000):
    a = torch.zeros((3,3))
    a = a.random_(generator = gen)
    ts.append(a)
ts = torch.stack(ts)
data_path = os.path.join('data', 'random.pt')
if os.path.exists(data_path):
    old = torch.load(data_path)
    print('all\n', (old == ts).all())
torch.save(ts, data_path)