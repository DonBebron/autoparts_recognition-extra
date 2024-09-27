import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_link):
    """
    Load an image from a given link or file path.
    
    Args:
        image_link (str or np.ndarray): The image source (URL, file path, or numpy array).
    
    Returns:
        PIL.Image.Image or None: The loaded image, or None if loading fails.
    """
    if type(image_link) == str:
        if image_link.startswith("http"):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                }
                response = requests.get(image_link, headers=headers)
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
    """
    Encode and normalize an image for model input.
    
    Args:
        img (PIL.Image.Image): The input image.
    
    Returns:
        np.ndarray: The encoded and normalized image array.
    """
    img = img.resize(cfg.image_size)
    img = np.array(img)
    img = img.astype('float32')
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-7
    img = (img + epsilon) / (255.0 + epsilon)
    
    # Clip values to ensure they're in the valid range [0, 1]
    img = np.clip(img, 0, 1)
    
    return img

def load_data(image_link):
    """
    Load and preprocess an image from a given link.
    
    Args:
        image_link (str): The URL or file path of the image.
    
    Returns:
        tf.Tensor or None: The preprocessed image tensor, or None if loading fails.
    """
    img = load_image(image_link)
    if img is None:
        return None
    img = encode_image(img)

    # convert data to tf.tensor
    img = tf.convert_to_tensor(img)
    return img

class Processor(metaclass=RuntimeMeta):
    """
    A class for processing web pages and images for model input.
    """
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    def get_page_content(self, url, verbose=0):
        """
        Retrieve and parse product information from a given URL.
        
        Args:
            url (str): The URL of the page to scrape.
            verbose (int): Verbosity level for logging.
        
        Yields:
            tuple: A pair of (image_src, product_link) for each product found.
        """
        logging.info(f"Getting page content from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to retrieve the webpage: {e}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try different selectors to find product items
        product_items = (
            soup.select('li.Product') or
            soup.select('div.ProductTile') or
            soup.select('div[class*="product"]')
        )
        
        if not product_items:
            logging.warning("No product items found on the page.")
        
        for item in product_items:
            link = item.select_one('a[href^="https://"]')
            img = item.select_one('img[src^="https://"]')
            if link and img:
                yield img.get('src'), link.get('href')
            else:
                logging.warning(f"Found incomplete product item: link={link}, img={img}")

    def parse_images_from_page(self, page_url):
        """
        Extract image links from a given page URL.
        
        Args:
            page_url (str): The URL of the page to parse.
        
        Returns:
            list: A list of unique image URLs found on the page.
        """
        logging.info(f"Parsing images from page: {page_url}")
        try:
            response = requests.get(page_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to retrieve the webpage: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try different selectors to find image elements
        image_elements = (
            soup.select('img.Product__imageData') or
            soup.select('img.ProductImage__image') or
            soup.select('img[src^="https://"]')
        )
        
        if not image_elements:
            logging.warning("No image elements found on the page.")
        
        image_links = []
        for img in image_elements:
            src = img.get('src')
            if src and src.startswith("https://"):
                image_links.append(src)
            else:
                logging.warning(f"Found invalid image source: {src}")
        
        unique_links = list(set(image_links))
        logging.info(f"Found {len(unique_links)} unique image links")
        return unique_links

    def load_product_info(self, url):
        """
        Load product information from a given URL.
        
        Args:
            url (str): The URL of the product page.
        
        Returns:
            dict: A dictionary containing product information (e.g., price).
        """
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
        """
        Build a TensorFlow dataset from a list of image links.
        
        Args:
            image_links (list): A list of image URLs or file paths.
        
        Returns:
            tf.data.Dataset: A TensorFlow dataset containing the processed images.
        """
        images = []
        for image_link in tqdm(image_links):
            img = load_data(image_link)
            if img is not None:
                images.append(img)

        if not images:
            print("Warning: No valid images found. Returning empty dataset.")
            return Dataset.from_tensor_slices([]).batch(1)  # Return an empty dataset

        dataset = Dataset.from_tensor_slices(images)
        
        # Add error checking
        try:
            # Check if the dataset is empty
            if tf.data.experimental.cardinality(dataset).numpy() == 0:
                print("Warning: Dataset is empty. Returning empty dataset.")
                return Dataset.from_tensor_slices([]).batch(1)
            
            # Try to fetch the first element
            next(iter(dataset))
        except Exception as e:
            print(f"Error in dataset: {e}")
            print("Returning empty dataset.")
            return Dataset.from_tensor_slices([]).batch(1)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __call__(self, *args, **kwargs):
        """
        Make the class callable, equivalent to calling build_dataset.
        """
        return self.build_dataset(*args, **kwargs)

    def take_newest(self, idx=10, *args, **kwargs):
        """
        Get the newest product page URL and parse its images.
        
        Args:
            idx (int): Index of the page to select (default is 10).
        
        Returns:
            list: A list of image URLs from the selected product page.
        """
        pages = [link for _, link in self.get_page_content(cfg.mainpage_url)]
        page_url = pages[idx] if idx < len(pages) else pages[-1]
        return self.parse_images_from_page(page_url)
