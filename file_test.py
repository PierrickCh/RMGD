import os
for f in os.listdir("data/saved_tensors"):
    c = [ i.split(".") for i in f.split("_")][-1][0] #retrieve hash, last list, first item
    
    print(c)