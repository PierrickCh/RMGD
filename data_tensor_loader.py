import os
import torch

import dataset_loader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##### MAIN PARAMETERS
""" Availible datasets :
celeba  (c)
cifar   (ci)
fashion (f)
art     (a)
cat     (ca)
dog     (d)
grumpy  (g)
obama   (o)
panda   (p)
flickr  (fl)
mnist   (m)
"""
def load_data_to_tensor(override_path:str, data_dir:str = "./data", data_set_name:str = "m", subset_name:str = None, dataset_loading_parameters:dict = None, force_reload_tensor=True) -> torch.Tensor:
    
    if override_path is not None and override_path != "" : 
        print(f"Loading tensor from : {override_path}")
        return torch.load(override_path, map_location=device)
    
    root_appendix = None # mainly used for parquet loading, specifies where to load under .data/ if specified

    # Handle assigning the right dataset number to each datasets and its corresponding loading function to be used
    assigned_data_set = -1
    loading_func = None

    match data_set_name:
        case "celeba"   | "c"  : assigned_data_set = 0     ;   loading_func = dataset_loader.load_celeba          ;    root_appendix = None
        case "cifar"    | "ci" : assigned_data_set = 1     ;   loading_func = dataset_loader.load_cifar10         ;    root_appendix = None
        case "fashion"  | "f"  : assigned_data_set = 2     ;   loading_func = dataset_loader.load_mnist_fashion   ;    root_appendix = None
        case "art"      | "a"  : assigned_data_set = 3     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-art-painting"
        case "cat"      | "ca" : assigned_data_set = 4     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-cat"
        case "dog"      | "d"  : assigned_data_set = 5     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-dog"
        case "grumpy"   | "g"  : assigned_data_set = 6     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-grumpy-cat"
        case "obama"    | "o"  : assigned_data_set = 7     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-obama"
        case "panda"    | "p"  : assigned_data_set = 8     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-panda"
        case "flickr"   | "fl" : assigned_data_set = 9     ;   loading_func = dataset_loader.load_jpg_folder      ;    root_appendix = None
        case "mnist"    | "m"  : assigned_data_set = 10    ;   loading_func = dataset_loader.load_mnist           ;    root_appendix = None

    full_name = f"{data_set_name}_{subset_name}_{assigned_data_set}"

    # loading/saving of cahce
    cache_dir = f"{data_dir}/saved_tensors"
    os.makedirs(cache_dir, exist_ok=True)

    status_file = os.path.join(cache_dir, "last_run_data.txt")
    tensor_file = os.path.join(cache_dir, f"tensor_cache_{full_name}.pt")

    
    load_from_cache = False
    if not force_reload_tensor and os.path.exists(status_file):
        load_from_cache = check_file_integrity("data/saved_tensors/last_run_data.txt",tensor_file,assigned_data_set,dataset_loading_parameters)

    if load_from_cache: # Load tensor from an already created cached tensor file
        print(f"Loading tensor from : {tensor_file}")
        data_tensor = torch.load(tensor_file, map_location=device)
    else: # Create a new cache tensor file
        print(f"Processing dataset {data_set_name} and creating tensor file ({tensor_file})")
        
        dataset_loading_parameters["root"] = os.path.join(data_dir, root_appendix) if root_appendix is not None else data_dir

        data_tensor = loading_func(**dataset_loading_parameters)
        # Save the tensor and update the file
        torch.save(data_tensor, tensor_file)
        with open(status_file, 'w') as f:
            f.write(f"{assigned_data_set}\n")
            for name, value in dataset_loading_parameters.items():
                f.write(f"{name}:{value}\n")

    return data_tensor

def check_file_integrity(last_run_path,tensor_file, current_data_set_id ,kwargs):
    """File structure : 
    dataset id (0-...)
    other parameters taken as a dictionnary, order should not matter with the check below
    """
    is_file_valid = False
    with open(last_run_path, 'r') as f:
        lines = [line.strip() for line in f.read().splitlines()]
        data_set_id = lines[0]
        if str(current_data_set_id) == str(data_set_id) and os.path.exists(tensor_file):
                is_file_valid = True
        parameters_saved:list = [i[0] for i in list(kwargs.items())]
        values_saved:list = [i[1] for i in list(kwargs.items())]
        for i,content in enumerate(lines[1:]) :
            parameter,value = content.split(":")
            
            if parameter in parameters_saved : 
                idx = parameters_saved.index(parameter)
                
                if str(value) != str(values_saved[idx]).replace(" ","") : 
                    
                    return False
        return is_file_valid
    

def save_dict_hash(d):
    from hashlib import sha256
    return sha256(str(d.items()).encode()).digest()