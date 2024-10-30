# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep
import random
import logging
import time
import json
import os
from PIL import Image
import requests
import re
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_PROMPT = """Identify the VAG (Volkswagen Audi Group) part number from the photo using this comprehensive algorithm:
1. **Scan the Image Thoroughly:**
   - Examine all text and numbers in the image, focusing on labels, stickers, or embossed areas.
   - Pay special attention to the upper part of labels, areas near barcodes, and any prominent alphanumeric sequences.
2. **Understand Detailed VAG Part Number Structure:**
   - Total length: Typically 11-13 characters (including spaces or hyphens)
   - Format: [First Number] [Middle Number] [Final Number] [Index] [Software Variant]
   
   Example: 5K0 937 087 AC Z15
   
   Detailed Breakdown:
   a) First Number (3 characters):
      - First two digits: Vehicle type (e.g., 3D = Phaeton, 1J = Golf IV, 8L = Audi A3)
      - Third digit: Body shape or variant
        0 = general, 1 = left-hand drive, 2 = right-hand drive, 3 = two-door, 4 = four-door,
        5 = notchback, 6 = hatchback, 7 = special shape, 8 = coupe, 9 = variant
   b) Middle Number (3 digits):
      - First digit: Main group (e.g., 1 = engine, 2 = fuel/exhaust, 3 = transmission, 4 = front axle, 5 = rear axle)
      - Last two digits: Subgroup within the main group
   c) Final Number (3 digits):
      - Identifies specific part within subgroup
      - Odd numbers often indicate left parts, even numbers right parts
   d) Index (1-2 LETTERS): Identifies variants, revisions, or colors
   e) Software Variant (2-3 characters): Often starts with Z (e.g., Z15, Z4)
3. **Identify and Verify with Precision:**
   - The first three parts (First, Middle, Final Numbers) are crucial and must be present.
   - Index and Software Variant may not always be visible or applicable.
   - Check for consistency with known vehicle types and component groups.
4. **Navigate Common Pitfalls and Special Cases:**
   - Character Confusion:
     '1' vs 'I', '0' vs 'O', '8' vs 'B', '5' vs 'S', '2' vs 'Z'
   - Upside-down numbers: Be vigilant for numbers that make sense when flipped.
   - Standard parts: May start with 9xx.xxx or 052.xxx
   - Exchange parts: Often marked with an 'X'
   - Color codes: e.g., GRU for primed parts requiring painting
5. **Context-Based Verification:**
   - Consider the part's apparent function in relation to its number.
   - Check for consistency with visible vehicle model or component type.
   - Look for supporting information like manufacturer logos or additional part descriptors.
Provide the response in this format:
- Valid part number identified: `<START> [VAG Part Number] <END>`
- No valid number found: `<START> NONE <END>`
Include spaces between number segments as shown in the example structure above.
If there are multiple numbers in the image, please identify the one that is most likely to be the correct part number.

**Response Format:**
- If a part number is identified: `<START> [Toyota Part Number] <END>`
- If no valid number is identified: `<START> NONE <END>`
"""

# Add token counting class
class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.image_tokens = 0  # New counter for image tokens
        
    def add_image_tokens(self):
        # According to Google's documentation, each image costs ~450 tokens
        IMAGE_TOKEN_COST = 450
        self.image_tokens += IMAGE_TOKEN_COST
        self.total_tokens += IMAGE_TOKEN_COST
        logging.info(f"Added {IMAGE_TOKEN_COST} tokens for image. Image tokens total: {self.image_tokens}")
        
    def add_tokens(self, response, prompt_tokens=0):
        try:
            # Count response tokens
            response_tokens = response.candidates[0].token_count
            self.response_tokens += response_tokens
            
            # Add prompt tokens
            self.prompt_tokens += prompt_tokens
            
            # Update totals
            self.total_tokens += (response_tokens + prompt_tokens)
            self.total_requests += 1
            
            logging.info(f"Request tokens - Prompt: {prompt_tokens}, Response: {response_tokens}")
            logging.info(f"Cumulative - Total: {self.total_tokens:,}, Requests: {self.total_requests}")
            
            return response_tokens
        except Exception as e:
            logging.warning(f"Could not count tokens: {e}")
            return 0
            
    def get_stats(self):
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "image_tokens": self.image_tokens,  # Added to stats
            "total_requests": self.total_requests,
            "average_tokens_per_request": self.total_tokens / self.total_requests if self.total_requests > 0 else 0
        }

class GeminiInference():
  def __init__(self, api_keys, model_name='gemini-1.5-flash', car_brand=None):
    self.api_keys = api_keys
    self.current_key_index = 0
    self.car_brand = car_brand.lower() if car_brand else None
    self.prompts = self.load_prompts()

    self.configure_api()
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
    ]

    # Get system prompt
    self.system_prompt = self.prompts.get(self.car_brand, {}).get('main_prompt', DEFAULT_PROMPT)
    
    # Initialize model with system prompt
    self.model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=self.system_prompt
    )

    # Count system prompt tokens
    try:
        system_tokens = len(self.system_prompt.split())  # Rough estimation
        logging.info(f"System prompt estimated tokens: {system_tokens}")
    except Exception as e:
        system_tokens = 0
        logging.warning(f"Could not estimate system prompt tokens: {e}")

    # Initialize token counter with system prompt
    self.token_counter = TokenCounter()
    self.token_counter.prompt_tokens += system_tokens
    self.token_counter.total_tokens += system_tokens

    self.validator_model = self.create_validator_model(model_name)
    self.incorrect_predictions = []
    self.message_history = []

  def load_prompts(self):
    try:
      with open('prompts.json', 'r') as f:
        return json.load(f)
    except FileNotFoundError:
      logging.warning("prompts.json not found. Using default prompts.")
      return {}

  def configure_api(self):
    genai.configure(api_key=self.api_keys[self.current_key_index])

  def switch_api_key(self):
    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
    self.configure_api()
    logging.info(f"Switched to API key index: {self.current_key_index}")

  def create_validator_model(self, model_name):
    # Configure the validator model with its own API key
    genai.configure(api_key=self.api_keys[self.current_key_index])
    
    # Create a separate model instance for validation
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
    ]
    return genai.GenerativeModel(model_name=model_name,
                                 generation_config=generation_config,
                                 safety_settings=safety_settings)

  def get_response(self, img_data, retry=False):
    max_retries = 10
    base_delay = 5  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            image_parts = [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_data.getvalue() if isinstance(img_data, io.BytesIO) else img_data.read_bytes()
                    }
                },
            ]
            
            prompt_parts = [] if not retry else [
                "It is not correct. Try again. Look for the numbers that are highly VAG number"
            ]
            
            full_prompt = image_parts + prompt_parts
            
            # Add a small random delay before each request
            sleep(random.uniform(1, 3))
            
            # Estimate prompt tokens (rough estimation)
            prompt_tokens = len(str(prompt_parts).split()) if retry else 0
            
            chat = self.model.start_chat(history=self.message_history)
            response = chat.send_message(full_prompt)
            
            # Count tokens including prompt
            self.token_counter.add_tokens(response, prompt_tokens)
            
            logging.info(f"Main model response: {response.text}")
            
            # Update message history
            self.message_history.append({"role": "user", "parts": full_prompt})
            self.message_history.append({"role": "model", "parts": [response.text]})
            
            return response.text
        except Exception as e:
            if "quota" in str(e).lower():
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                if delay > 300:  # If delay is more than 5 minutes
                    self.switch_api_key()
                    delay = base_delay  # Reset delay after switching
                logging.warning(f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Error in get_response: {str(e)}")
                raise
    
    logging.error("Max retries reached. Unable to get a response.")
    raise Exception("Max retries reached. Unable to get a response.")

  def format_part_number(self, number):
    # Check if the number matches the VAG (Audi) pattern
    if self.car_brand == 'audi' and re.match(r'^[A-Z0-9]{3}[0-9]{3}[0-9]{3,5}[A-Z]?$', number.replace(' ', '').replace('-', '')):
        # Remove any existing hyphens and spaces
        number = number.replace('-', '').replace(' ', '')
        
        # Format the first 9 characters
        formatted_number = f"{number[:3]} {number[3:6]} {number[6:9]}"
        
        # Add the remaining characters, if any
        if len(number) > 9:
            formatted_number += f" {number[9:]}"

        return formatted_number.strip()
    else:
        # If it's not an Audi number or doesn't match the pattern, return the original number
        return number

  def extract_number(self, response):
    number = response.split('<START>')[-1].split("<END>")[0].strip()
    if number.upper() != "NONE":
      return self.format_part_number(number)
    return number

  def validate_number(self, extracted_number, img_data):
    # Ensure we're using the validator API key for this request
    genai.configure(api_key=self.api_keys[self.current_key_index])
    
    # Format the extracted number before validation
    formatted_number = self.format_part_number(extracted_number)
    
    image_parts = [
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_data.getvalue() if isinstance(img_data, io.BytesIO) else img_data.read_bytes()
            }
        },
    ]
    
    # Count image tokens for validation
    self.token_counter.add_image_tokens()
    
    validation_prompt = self.prompts.get(self.car_brand, {}).get('validation_prompt', "")
    incorrect_predictions_str = ", ".join(self.incorrect_predictions)
    prompt = validation_prompt.format(extracted_number=extracted_number, incorrect_predictions=incorrect_predictions_str)

    prompt_parts = [
        image_parts[0],
        prompt,
    ]

    # Estimate validation prompt tokens
    prompt_tokens = len(prompt.split())  # Rough estimation
    
    response = self.validator_model.generate_content(prompt_parts)
    
    # Count tokens including validation prompt
    self.token_counter.add_tokens(response, prompt_tokens)
    
    logging.info(f"Validator model response: {response.text}")
    return response.text

  def reset_incorrect_predictions(self):
    self.incorrect_predictions = []
    self.message_history = []  # Reset message history along with incorrect predictions

  def __call__(self, image_path):
    self.configure_api()  # Ensure we're using the current API key
    
    # Handle remote or local images
    if image_path.startswith('http'):
        # Read remote image into memory
        response = requests.get(image_path, stream=True)
        img_data = io.BytesIO(response.content)
    else:
        # Local file path
        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(f"Could not find image: {img}")
        img_data = img

    # Reset message history for new image
    self.message_history = []

    max_attempts = 2  # Initial attempt + 1 additional attempt
    for attempt in range(max_attempts):
        # Generate response and extract number
        answer = self.get_response(img_data, retry=(attempt > 0))
        extracted_number = self.extract_number(answer)
        
        logging.info(f"Attempt {attempt + 1}: Extracted number: {extracted_number}")
        
        if extracted_number.upper() != "NONE":
            validation_result = self.validate_number(extracted_number, img_data)
            if "<VALID>" in validation_result:
                logging.info(f"Valid number found: {extracted_number}")
                self.reset_incorrect_predictions()
                return extracted_number
            else:
                logging.warning(f"Validation failed: {validation_result}")
                self.incorrect_predictions.append(extracted_number)
                if attempt < max_attempts - 1:
                    logging.info(f"Attempting to find another number (Attempt {attempt + 2}/{max_attempts})")
        else:
            logging.warning(f"No number found in attempt {attempt + 1}")
            if attempt < max_attempts - 1:
                logging.info(f"Attempting to find another number (Attempt {attempt + 2}/{max_attempts})")

    logging.warning("All attempts failed. Returning NONE.")
    self.reset_incorrect_predictions()  # Reset for the next page
    return "NONE"

  def get_token_stats(self):
    """Get statistics about token usage"""
    return self.token_counter.get_stats()
