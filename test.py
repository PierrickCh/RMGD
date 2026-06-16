
import hashlib
d:dict = {
    "num_samples" : 250,
    "target_labels" : [] ,
    "image_size" : 32,  
    
      
}

print(hashlib.sha256(str(d.items()).encode()).digest())