# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep
import random
import logging

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

class GeminiInference():
  def __init__(self, api_key, validator_api_key, model_name='gemini-1.5-flash', prompt=None):
    self.gemini_key = api_key
    self.validator_key = validator_api_key
    self.prompt = prompt if prompt is not None else DEFAULT_PROMPT

    genai.configure(api_key=self.gemini_key)
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    self.model = genai.GenerativeModel(model_name=model_name,
                                       generation_config=generation_config,
                                       safety_settings=safety_settings,
                                       system_instruction=self.prompt)

    self.validator_model = self.create_validator_model(model_name)
    self.incorrect_predictions = []
    self.message_history = []

  def create_validator_model(self, model_name):
    # Configure the validator model with its own API key
    genai.configure(api_key=self.validator_key)
    
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
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]
    return genai.GenerativeModel(model_name=model_name,
                                 generation_config=generation_config,
                                 safety_settings=safety_settings)

  def get_response(self, img_data, retry=False):
    max_retries = 20
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
            
            chat = self.model.start_chat(history=self.message_history)
            response = chat.send_message(full_prompt)
            logging.info(f"Main model response: {response.text}")
            
            # Update message history
            self.message_history.append({"role": "user", "parts": full_prompt})
            self.message_history.append({"role": "model", "parts": [response.text]})
            
            return response.text
        except Exception as e:
            if "quota" in str(e).lower():
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f} seconds...")
                sleep(delay)
            else:
                logging.error(f"Error in get_response: {str(e)}")
                raise  # Re-raise the exception if it's not a rate limit error
    
    logging.error("Max retries reached. Unable to get a response.")
    raise Exception("Max retries reached. Unable to get a response.")

  def format_part_number(self, number):
    # Remove any existing hyphens and spaces
    number = number.replace('-', '').replace(' ', '')
    
    # Ensure the number has at least 9 characters
    if len(number) < 9:
        return number  # Return original if too short
    
    # Format the first 9 characters
    formatted_number = f"{number[:3]} {number[3:6]} {number[6:9]}"
    
    # Add the remaining characters, if any
    if len(number) > 9:
        formatted_number += f" {number[9:]}"

    return formatted_number.strip()

  def extract_number(self, response):
    number = response.split('<START>')[-1].split("<END>")[0].strip()
    if number.upper() != "NONE":
      return self.format_part_number(number)
    return number

  def validate_number(self, extracted_number, img_data):
    # Ensure we're using the validator API key for this request
    genai.configure(api_key=self.validator_key)
    
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
    
    prompt = f"""
     Validate the following VAG (Volkswagen Audi Group) part number: {extracted_number}
    Rules for validation:
    1. The number should consist of 9-11 characters.
    2. It may or may not be visibly divided into groups.
    3. The structure should closely follow this pattern:
       - First part: 3 characters (e.g., "5Q0", "8S0")
       - Middle part: 3 digits (e.g., "937", "907")
       - Last part: 3-4 characters, which may include digits and/or letters (e.g., "085B", "468D")
    4. The entire number may be continuous without spaces, but should still follow the above structure.
    5. Pay extra attention to commonly confused digits:
       - '9' and '8' can be easily confused
       - '0' and 'O' (letter O) should not be mixed up
       - '1' and 'I' (letter I) should not be confused
    6. The last part should not contain any digits after known letter suffixes (e.g., "AD" should not be followed by digits)
    7. If the last part ends with a single letter, make sure it's not missing (e.g., "T" at the end)
    8. Ensure no extra digits or characters are included that don't belong to the actual part number.
    9. Check if the number could be an upside-down non-VAG number:
       - Look for patterns that might make sense when flipped (e.g., "HOSE" could look like "3SOH" upside down)
       - Be cautious of numbers that don't follow the typical VAG format but could be valid when flipped

       Previously incorrect predictions on this page: {', '.join(self.incorrect_predictions)}

       Try to think step by step, do not rush.

    If the number follows these rules and is not likely to be an upside-down non-VAG number, respond with:
    <VALID>
    If the number does not follow these rules, seems incorrect, or could be an upside-down non-VAG number, respond with:
    <INVALID>
    If the number does not follow these rules at all, respond with (in the explanation ask model to look for another line in the upper right corner of the label that might contain the part number (which is usually bigger)):
    <INVALID>
    Explanation: [Brief explanation of why it's valid or invalid, including the number itself and any concerns about it being upside-down]
    """

    prompt_parts = [
        image_parts[0],
        prompt,
    ]

    response = self.validator_model.generate_content(prompt_parts)
    logging.info(f"Validator model response: {response.text}")
    return response.text

  def reset_incorrect_predictions(self):
    self.incorrect_predictions = []
    self.message_history = []  # Reset message history along with incorrect predictions

  def __call__(self, image_path):
    # Ensure we're using the main API key for the main model
    genai.configure(api_key=self.gemini_key)
    
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

    max_attempts = 3  # Initial attempt + 2 additional attempts
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
                    logging.info(f"Attempting to find another VAG number (Attempt {attempt + 2}/{max_attempts})")
        else:
            logging.warning(f"No number found in attempt {attempt + 1}")
            if attempt < max_attempts - 1:
                logging.info(f"Attempting to find another VAG number (Attempt {attempt + 2}/{max_attempts})")

    logging.warning("All attempts failed. Returning NONE.")
    self.reset_incorrect_predictions()  # Reset for the next page
    return "NONE"
