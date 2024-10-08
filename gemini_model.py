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

Please follow the above steps to recognize the correct detail number and format the response as follows:

**Response Format:**
- If a part number is identified: `<START> [VAG Part Number] <END>`
- If no valid number is identified: `<START> NONE <END>`
"""

class GeminiInference():
  def __init__(self, api_key, model_name='gemini-1.5-flash', prompt=None):
    self.gemini_key = api_key
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
    # Create a separate model instance for validation
    generation_config = {
        "temperature": 0.5,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        # ... (same safety settings as the main model)
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
                "It is not correct. Try again."
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
    Strictly validate the following VAG (Volkswagen Audi Group) part number: {formatted_number}

    Examine the provided image meticulously and confirm that the extracted number {formatted_number} is EXACTLY and CLEARLY visible in the image. Look for this precise sequence of characters, focusing on labels, stickers, or embossed areas.

    Validation Rules (ALL MUST BE MET for a VALID result):

    1. Exact Match: The number in the image must EXACTLY match {formatted_number}, including all spaces and characters.
    2. Clear Visibility: The number must be clearly readable and not obscured or partially visible.
    3. Correct Location: Typically found on labels, near barcodes, or embossed on the part itself.
    4. Proper Format: Must strictly follow the pattern [First Number] [Middle Number] [Final Number] [Index] [Software Variant]
       Example: 5K0 937 087 AC Z15
       
    5. Component Breakdown (each part must be correct):
       a) First Number (3 characters):
          - First two digits: Valid vehicle type code
          - Third digit: Valid body shape or variant code (0-9)
       b) Middle Number (3 digits):
          - First digit: Valid main group code (0-9)
          - Last two digits: Valid subgroup code
       c) Final Number (3 digits):
          - Must be a valid specific part identifier
       d) Index (if present): 1-2 letters only
       e) Software Variant (if present): Must start with Z followed by numbers

    6. No Digit-Letter Confusion:
       - '1' and 'I', '0' and 'O', '8' and 'B', '5' and 'S', '2' and 'Z' must not be confused
    7. No Upside-Down Numbers: Ensure the number isn't an inverted non-VAG number
    8. Special Cases (if applicable):
       - Exchange parts: Marked with 'X' at the end
       - Standard parts: May start with 9xx.xxx or 052.xxx
       - Specific prefixes: 'G' for lubricants, 'D' for sealants, 'B' for brake fluids

    Previously incorrect predictions: {', '.join(self.incorrect_predictions)}

    Based on your strict examination, respond ONLY with one of the following:

    If ALL validation rules are met and the exact number is found:
    <VALID>

    If the exact number is found but ANY rule is violated:
    <INVALID>

    If the exact number is not found in the image:
    <NOT_FOUND>

    Explanation: [Concise explanation of your findings, including specific reasons for invalidity or not found status]
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

    # Generate response and extract number
    answer = self.get_response(img_data)
    extracted_number = self.extract_number(answer)
    
    logging.info(f"Extracted number: {extracted_number}")
    
    if extracted_number.upper() != "NONE":
        validation_result = self.validate_number(extracted_number, img_data)
        if "<VALID>" in validation_result:
            logging.info(f"Valid number found: {extracted_number}")
            self.reset_incorrect_predictions()
            return extracted_number
        elif "<NOT_FOUND>" in validation_result:
            logging.warning(f"Extracted number not found in image: {validation_result}")
            self.incorrect_predictions.append(extracted_number)
        else:
            logging.warning(f"Validation failed: {validation_result}")
            self.incorrect_predictions.append(extracted_number)
    else:
        logging.warning("No number found")
        
        # If NONE is returned, try one more time with the same prompt
        logging.info("Attempting one more time with the same image")
        
        retry_answer = self.get_response(img_data, retry=True)  # This will use the retry prompt and message history
        retry_extracted_number = self.extract_number(retry_answer)
        
        logging.info(f"Retry attempt: Extracted number: {retry_extracted_number}")
        
        if retry_extracted_number.upper() != "NONE":
            validation_result = self.validate_number(retry_extracted_number, img_data)
            if "<VALID>" in validation_result:
                logging.info(f"Valid number found in retry: {retry_extracted_number}")
                self.reset_incorrect_predictions()
                return retry_extracted_number
            else:
                logging.warning(f"Retry validation failed: {validation_result}")
                self.incorrect_predictions.append(retry_extracted_number)
        else:
            logging.warning("No number found in retry attempt")

    logging.warning("All attempts failed. Returning NONE.")
    self.reset_incorrect_predictions()  # Reset for the next page
    return "NONE"
