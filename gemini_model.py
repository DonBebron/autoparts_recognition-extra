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

DEFAULT_PROMPT = """     identify Main Catalog Number from photo by this Algorithm


**Note:** This algorithm is specifically designed to identify and verify VAG part numbers. Assume all part numbers belong to VAG by default.

1. **Identify Potential Numbers on the Photo:**
   - **Examine All Numbers:** Carefully inspect the image for various numbers and markings.
   - **Identify Numbers Resembling Catalog Numbers:** Look for numbers with a structured format that includes a combination of digits and letters, often separated into groups (e.g., `1K2 820 015 C`). This format is typical for parts catalog numbers.
   - **Clarification:** Focus on numbers that fit the standard format (three groups of characters) and avoid mistaking additional characters or version codes (e.g., "H03" or "0012") as part of the primary catalog number. 

2. **Analyze the Structure of the Numbers:**
   - **Catalog Numbers (Part Numbers):** These numbers typically consist of 9-10 characters, divided into groups, each carrying specific information:
     - **First Group:** Indicates the car model or platform (e.g., "1K2" for VW Golf V).
     - **Second Group:** Describes the type of part (e.g., "820" for air conditioning systems).
     - **Third Group:** Refers to the version or modification of the part.
   - **OEM Numbers:** OEM (Original Equipment Manufacturer) numbers may not follow these rules, often start with letters, have a less structured appearance, and may be accompanied by the manufacturer's logo.
   - **Clarification:** Avoid including version or revision codes (e.g., "H03", "0012") within the primary catalog number unless it's specifically relevant to the core part number format.

3. **Determine the Brand by the Numbers:**
   - **VAG Numbers:** Numbers that follow the Volkswagen Audi Group (VAG) standard adhere to the structure described above. These numbers often do not have logos from third-party manufacturers.
   - **OEM Numbers:** If the number is accompanied by a third-party manufacturer's logo (e.g., Valeo), and the number does not follow the standard VAG format, it is likely an OEM number.
   - **Clarification:** Prioritize the identification of the main VAG catalog number (e.g., "4H0 907 801 E") and treat any additional characters (e.g., "0012", "H03") as supplementary information, not as part of the main catalog number.

4. **Verify the Accuracy of the Number:**
   - **Match with Known Formats:** Compare the identified number with commonly accepted formats for the brand. For example, VAG numbers should adhere to the standard format described above.
   - **Check Against Catalogs:** If possible, use an official parts catalog to verify the number. Enter the number into the catalog's search system to ensure it corresponds to the correct part.
   - **Compare with Other Numbers:** If multiple numbers are present on the part, ensure that the number you have identified matches the described format and is the primary catalog number, not the OEM number.
   - **Clarification:** When comparing numbers, ensure that you separate the core catalog number from any additional version codes or supplementary characters.

5. **Final Check and Marking:**
   - **Absence of Third-Party Logos:** Ensure that the number you believe to be the catalog number is not accompanied by a third-party manufacturer's logo (if it is supposed to be an original VAG number).
   - **Logical Placement:** The catalog number is usually placed in a prominent location or on the main part of the label, making it easier to identify.
   - **Clarification:** Focus on the number in the main location (often near the brand's logo) as the primary catalog number and treat additional codes as supplementary, ensuring they don't replace the main number.


Please follow the above steps to recognize the correct detail number and format the response as follows:

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
