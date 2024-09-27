from dataprocessor import * 
from config import * 

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def build_model(num_classes) -> Model:
    """
    Builds a small image classifier using MobileNetV3Small backbone.

    Parameters:
      optimizer: AdamW
      loss: binary_crossentropy
      metrics: accuracy

    Args:
      num_classes: Number of classes for classification.

    Returns:
      A Keras model.
    """
    # Load pre-trained MobileNetV3Small model (without top layers)
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='gelu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# model = build_model(1)
# model.load_weights(cfg.model_path)

class TargetModel(metaclass=RuntimeMeta):
  def __init__(self, model_path = None):
    # self.gemini = GeminiInference()
    if model_path == None: 
      model_path = cfg.model_path


    self.model = build_model(1)
    self.model.load_weights(model_path)

    self.processor = Processor(cfg.image_size, cfg.batch_size)

    self.predicted_image_saving_path = "example_prediction.jpg"

  def do_inference_return_probs(self, image_links): 
    dataset = self.processor(image_links)
    predictions = self.model.predict(dataset)

    predictions = predictions.flatten().tolist() 
    predictions = [{'image_link': l, 'score': p} for l, p in zip(image_links, predictions)]
    return sorted(predictions, key=lambda i: float(i['score']), reverse=True)

  def do_inference_minimodel(self, *args, **kwargs):
    results = self.do_inference_return_probs(*args, **kwargs)
    return results[0]['image_link']

  def do_inference(self, image_links, *args, **kwargs):
    target_image_link = self.do_inference_minimodel(image_links, *args, **kwargs)

    # save target_image_link to local image if it link. return local path

    if (target_image_link.startswith("http")):
      response = requests.get(target_image_link)
      img = Image.open(BytesIO(response.content))
      img.save(self.predicted_image_saving_path)
      target_image_link = self.predicted_image_saving_path

    print(f'target image link: {target_image_link}')

    # predicted, image_saved_path = self.gemini(target_image_link), target_image_link

    return target_image_link

  def __call__(self, *args, **kwargs):
    return self.do_inference(*args, **kwargs)

  def predict_newest(self, idx=10, *args, **kwargs):
    return self.do_inference(self.processor.take_newest(idx), *args, **kwargs)

# do_inference = TargetModel(model)


# # predictions, groups = do_inference.predict_newest()

# image_links = do_inference.processor.parse_images_from_page('https://injapan.ru/auction/v1114858293.html')
# predictions, groups = do_inference(image_links)
# print("PREDICTED :", predictions)
# l = logs.pop()