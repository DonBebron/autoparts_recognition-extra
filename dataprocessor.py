from config import Config as cfg 
from config import RuntimeMeta

import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tqdm.notebook import tqdm

import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

def load_image(image_link):
    if type(image_link) == str:
      if image_link.startswith("http"):
          try:
              headers = {
                  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
              }

              response = requests.get(image_link, headers = headers)
              # response = requests.get(image_link
              img = Image.open(BytesIO(response.content))
          except Exception as e:
              print(image_link)
              print(e)
              return None
      else:
          img = Image.open(image_link)
    elif type(image_link) == np.ndarray:
      img = Image.fromarray(image_link)
    else:
      raise Exception("Unknown image type")

    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def encode_image(img):
    img = img.resize(cfg.image_size)
    img = np.array(img)
    img = img.astype('float32')
    img /= 255.0
    return img

def load_data(image_link):
    img = load_image(image_link)
    if img is None:
        return None
    img = encode_image(img)

    # convert data to tf.tensor
    img = tf.convert_to_tensor(img)
    return img


class Processor(metaclass=RuntimeMeta):
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    def get_page_content(self, url, verbose=0):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find all product items on the Yahoo Auctions page
            product_items = soup.find_all('li', class_='Product')
            
            for item in product_items:
                # Extract the product link and image source for each item
                link = item.find('a', class_='Product__imageLink')
                img = item.find('img', class_='Product__imageData')
                if link and img:
                    yield img.get('src'), link.get('href')
        else:
            print(f'Failed to retrieve the webpage. Status code: {response.status_code}')

    def parse_images_from_page(self, page_url):
        # Extract all image links from the product page
        image_links = [i for i, _ in self.get_page_content(page_url) if i and i.startswith("https://")]
        return list(set(image_links))

    def load_product_info(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return_data = {}
            # Extract the price information from the product page
            price_elem = soup.find('dd', class_='Price__value')
            return_data['price'] = price_elem.text.strip() if price_elem else 'N/A'
            return return_data
        else:
            print(f'Failed to retrieve the webpage. Status code: {response.status_code}')

    def build_dataset(self, image_links):
        images = [load_data(image_link) for image_link in tqdm(image_links)]
        images = [img for img in images if img is not None]

        dataset = Dataset.from_tensor_slices(images)
        try:
            next(iter(dataset))
        except Exception as e:
            print(e)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __call__(self, *args, **kwargs):
        return self.build_dataset(*args, **kwargs)

    def take_newest(self, idx=10, *args, **kwargs):
        pages = [link for _, link in self.get_page_content(cfg.mainpage_url)]
        page_url = pages[idx] if idx < len(pages) else pages[-1]
        return self.parse_images_from_page(page_url)
