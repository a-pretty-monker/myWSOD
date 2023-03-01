from GAT import GAT_NET


model = GAT_NET(4096, 2048,20)
for name in model.state_dict():
  print(name)

