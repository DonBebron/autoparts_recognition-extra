import logging
from config import Config as cfg 
from config import RuntimeMeta

import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset

import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import time
import random
import re
from requests.exceptions import RequestException, ProxyError

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
        self.session = requests.Session()
        self.user_agents = self.generate_similar_user_agents()
        self.headers_list = self.generate_headers_list()
        self.proxies = [
            # Add your list of proxy servers here, for example:
            # {'http': 'http://10.10.1.10:3128', 'https': 'http://10.10.1.10:1080'},
            # {'http': 'http://10.10.1.11:3128', 'https': 'http://10.10.1.11:1080'},
        ]

    def generate_similar_user_agents(self):
        """Return a list of highly reliable PC-like user agents that are widely accepted."""
        return [
            # Chrome on Windows - Most common and widely accepted
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            
            # Firefox on Windows - Second most common
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
            
            # Edge on Windows - Microsoft's default browser
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
            
            # Chrome on macOS - Most common macOS browser
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            
            # Safari on macOS - Default macOS browser
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15"
        ]

    def generate_headers_list(self):
        """Generate a list of headers with different user agents and additional variations."""
        base_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://auctions.yahoo.co.jp/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        headers_list = []
        for user_agent in self.user_agents:
            headers = base_headers.copy()
            headers['User-Agent'] = user_agent
            
            # Add some randomness to the headers
            if random.random() < 0.5:
                headers['Accept-Language'] = random.choice(['en-US,en;q=0.9', 'en-GB,en;q=0.8,en-US;q=0.6', 'en-CA,en-US;q=0.9,en;q=0.8'])
            if random.random() < 0.3:
                headers['Sec-Ch-Ua'] = '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"'
            if random.random() < 0.3:
                headers['Sec-Ch-Ua-Mobile'] = '?0'
            if random.random() < 0.3:
                headers['Sec-Ch-Ua-Platform'] = '"Windows"'
            
            headers_list.append(headers)
        return headers_list

    def get_page_content(self, url, verbose=0, max_retries=5):
        """
        Retrieve and parse product information from a given URL.
        
        Args:
            url (str): The URL of the page to scrape.
            verbose (int): Verbosity level for logging.
            max_retries (int): Maximum number of retry attempts.
        
        Yields:
            tuple: A pair of (image_src, product_link) for each product found.
        """
        logging.info(f"Getting page content from: {url}")

        for attempt in range(max_retries):
            headers = random.choice(self.headers_list)
            headers['User-Agent'] = random.choice(self.user_agents)  # Use the new method here
            
            try:
                delay = (2 ** attempt) + random.random()
                time.sleep(delay)
                
                response = self.session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
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
                
                # If we've successfully processed the page, break the retry loop
                break
            
            except RequestException as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                logging.error(f"Headers used: {headers}")
                if attempt == max_retries - 1:
                    logging.error(f"Failed to retrieve the webpage after {max_retries} attempts: {e}")
                    return

    def parse_images_from_page(self, page_url, max_retries=5):
        """
        Extract image links from a given page URL, focusing on the "ProductImage__images" class.
        
        Args:
            page_url (str): The URL of the page to parse.
        
        Returns:
            list: A list of unique image URLs found within the "ProductImage__images" class.
        """
        logging.info(f"Parsing images from page: {page_url}")

        for attempt in range(max_retries):
            headers = random.choice(self.headers_list)
            
            try:
                time.sleep(random.uniform(1, 2))
                response = self.session.get(page_url, headers=headers, timeout=15)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                logging.error(f"Headers used: {headers}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.random()
                    logging.warning(f"Request failed. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to retrieve the webpage after {max_retries} attempts: {e}")
                    return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        image_links = []

        # Find images in the "ProductImage__images" class
        product_image_elements = soup.find_all(class_="ProductImage__images")
        
        if product_image_elements:
            for element in product_image_elements:
                imgs = element.find_all('img')
                for img in imgs:
                    src = img.get('src') or img.get('data-src')
                    if src:
                        image_links.append(src)
            logging.info(f"Found {len(image_links)} images")
        else:
            logging.warning("No 'ProductImage__images' class found on the page")

        # Process and clean up image links
        cleaned_links = []
        for src in image_links:
            if src.startswith('//'):
                src = 'https:' + src
            elif not src.startswith('http'):
                src = 'https://auctions.yahoo.co.jp' + src
            cleaned_links.append(src)

        unique_links = list(set(cleaned_links))
        logging.info(f"Found {len(unique_links)} unique image links")
        
        if not unique_links:
            logging.warning("No images found. Dumping HTML for inspection.")
            with open('page_dump.html', 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            logging.warning("HTML dumped to page_dump.html")
        
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
        for i, image_link in enumerate(image_links):
            img = load_data(image_link)
            if img is not None:
                images.append(img)
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{len(image_links)} images")

        if not images:
            logging.warning("No valid images found. Returning empty dataset.")
            return Dataset.from_tensor_slices([]).batch(1)  # Return an empty dataset

        dataset = Dataset.from_tensor_slices(images)
        
        # Add error checking
        try:
            # Check if the dataset is empty
            if tf.data.experimental.cardinality(dataset).numpy() == 0:
                logging.warning("Dataset is empty. Returning empty dataset.")
                return Dataset.from_tensor_slices([]).batch(1)
            
            # Try to fetch the first element
            next(iter(dataset))
        except Exception as e:
            logging.error(f"Error in dataset: {e}")
            logging.warning("Returning empty dataset.")
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
