









kwargs = {
    "num_samples" : 150,
    "target_labels" : [5,6,9],
    "image_size" : 32,    
}







with open("data/saved_tensors/last_run_data.txt", 'r') as f:
    lines = [line.strip() for line in f.read().splitlines()]
    data_set_id = int(lines[0])
    parameters_saved:list = [i[0] for i in list(kwargs.items())]
    values_saved:list = [i[1] for i in list(kwargs.items())]
    for i,content in enumerate(lines[1:]) :
        parameter,value = content.split(":")
        
        if parameter in parameters_saved : 
            idx = parameters_saved.index(parameter)
            print(value, values_saved[idx])
            if str(value) != str(values_saved[idx]).replace(" ","") : 
                print("r - must reload tensor")

    print("r - load from cache")