# lib data manipulation 
import pandas as pd

# function load dataset
def data_hotspot():

  # load dataset
  dataset = pd.read_csv("dataset/dataset_hotspot.csv", parse_dates=['acq_date'])
  
  # return values
  return dataset
