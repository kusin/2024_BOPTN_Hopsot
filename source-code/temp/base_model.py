from class_model import loadData

# ALGORITMA
if __name__ == "__main__":

    # set dataset name
    df_name = "dataset_hotspot.csv"
    
    # load dataset
    dataset = loadData(df_name)
    print(dataset)