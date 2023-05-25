from preprocessing import *


data = load_csv(nrows=50)
data_prepared = prepare(data)
target = extract_target(data)