import json
pos={"mean":0.15195166170597077,"std":0.007944623319042856}
raw={"mean":6.596016028197482e-05,"std":1.4928941079651003e-05}
wrong={"mean":0.0031007011188194157,"std":0.00039428395572848305}
cand=max(raw["mean"]+6*raw["std"], wrong["mean"]+6*wrong["std"])
T=max(0.02,cand)
print(json.dumps({"suggested_threshold":T},indent=2))
