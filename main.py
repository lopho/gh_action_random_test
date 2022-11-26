import torch
import os

torch.manual_seed(0)
ts = []
for i in range(1000):
    a = torch.randn((3,3))
    ts.append(a)
ts = torch.stack(ts)
data_path = os.path.join('data', 'random.pt')
old_data = os.path.exists(data_path)
old = torch.load(data_path) if old_data else None
torch.save(ts, data_path)
if old_data:
    print('all\n', (old == ts).all())
    assert (old == ts).all()
