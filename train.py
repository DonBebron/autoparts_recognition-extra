from dataprocessor import load_data
from tensorflow.data import Dataset 
import tensorflow as tf

tf.executing_eagerly()


from picker_model import TargetModel 

def image_mapping_fn(image_link): 
  image_link = bytes.decode(image_link.numpy())
  # image_link = bytes.decode(image_link)

  return load_data(image_link)

class Trainer(TargetModel): 
  def __init__(self, 
               dataset = None, 
               dataset_path=None): 
    super().__init__()
    
    if dataset == None: 
      dataset = self.read_from_dataset_path(dataset_path)
    self.dataset = dataset

    self.dataset_dict = {} 
    for item in self.dataset: 
      self.dataset_dict = {**self.dataset_dict, **item}

  def read_from_dataset_path(self, dataset_folder): 
    json_files = []
    for root, dirs, files in os.walk(dataset_folder):
      for file in files:
        if file.endswith('.json'):
          json_files.append(os.path.join(root, file))

    data = []
    for json_file in json_files:
      with open(json_file, 'r') as f:
        data.append(json.load(f))
    return data 

  def build_dataset(self): 
    image_links = list(self.dataset_dict.keys())
    labels = list(self.dataset_dict.values()) 
    
    self.dataset = (Dataset.from_tensor_slices({"image": image_links, "l": labels})
    .map(lambda i: {"image": tf.py_function(image_mapping_fn, [(i['image'])], [tf.float32]), "label": i['l']}))
    return self.dataset.batch(128)

  def train(self): 
    dataset = self.build_dataset()

trainer = Trainer(dataset = correct_answers)
trainer.train()
