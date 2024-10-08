# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep
import random

from PIL import Image
import requests
import re

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
   d) Index (1-2 letters): Identifies variants, revisions, or colors
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
                                  safety_settings=safety_settings)

    self.validator_model = self.create_validator_model(model_name)
    self.incorrect_predictions = []

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

  def get_response(self, img):
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": img.read_bytes()
        },
    ]
    prompt_parts = [
        image_parts[0],
        (self.prompt if self.prompt is not None else "..."),  # Existing prompt
    ]
    
    max_retries = 20
    base_delay = 5  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Add a small random delay before each request
            sleep(random.uniform(1, 3))
            
            response = self.model.generate_content(prompt_parts)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower():
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f} seconds...")
                sleep(delay)
            else:
                raise  # Re-raise the exception if it's not a rate limit error
    
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

  def validate_number(self, extracted_number):
    prompt = f"""
    Validate the following VAG (Volkswagen Audi Group) part number: {extracted_number}

    First, carefully examine the image and confirm that the extracted number {extracted_number} is actually visible in the image. Look for this exact sequence of characters, paying attention to labels, stickers, or any printed/embossed areas.

    If you can find the exact number in the image, proceed with the following validation rules:

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

    Based on your examination of the image and the validation rules, respond with one of the following:

    If the exact number is found in the image and follows the rules:
    <VALID>

    If the exact number is found in the image but does not follow the rules:
    <INVALID>

    If the exact number is not found in the image:
    <NOT_FOUND>

    If the number does not follow these rules at all or is not found, also include in your explanation a suggestion to look for another line in the upper right corner of the label that might contain the correct part number (which is usually bigger).

    Explanation: [Brief explanation of your findings, including whether the number was found in the image, why it's valid or invalid, and any concerns about it being upside-down or misread]
    """

    response = self.validator_model.generate_content(prompt)
    return response.text

  def reset_incorrect_predictions(self):
    self.incorrect_predictions = []

  def __call__(self, image_path):
    # Validate that an image is present
    if image_path.startswith('http'):
        # read remote img bytes
        img = Image.open(requests.get(image_path, stream=True).raw)
        # save image to local "example_image.jpg"
        img.save("example_image.jpg")
        image_path = "example_image.jpg"

    if not (img := Path(image_path)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    max_attempts = 3
    original_prompt = self.prompt

    for attempt in range(max_attempts):
        # Generate response and extract number
        answer = self.get_response(img)
        extracted_number = self.extract_number(answer)
        
        if extracted_number.upper() != "NONE":
            validation_result = self.validate_number(extracted_number)
            if "<VALID>" in validation_result:
                self.reset_incorrect_predictions()
                self.prompt = original_prompt  # Reset prompt to original
                return extracted_number
            elif "<NOT_FOUND>" in validation_result:
                print(f"Extracted number not found in image (Attempt {attempt + 1}): {validation_result}")
                self.incorrect_predictions.append(extracted_number)
            else:
                print(f"Validation failed (Attempt {attempt + 1}): {validation_result}")
                self.incorrect_predictions.append(extracted_number)
        else:
            print(f"No number found (Attempt {attempt + 1})")

        # If this is not the last attempt, create a more specific prompt for the next try
        if attempt < max_attempts - 1:
            specific_prompt = f"""
            {original_prompt}

            Additional instructions for attempt {attempt + 2}:
            {f'The previously extracted number "{extracted_number}" was invalid.' if extracted_number.upper() != "NONE" else "No valid number was found in the previous attempt(s)."}
            Previously incorrect predictions on this page: {', '.join(self.incorrect_predictions)}

            Please re-examine the image carefully and try to identify a valid VAG part number.
            Focus on the following:
            1. Look for numbers that are larger or more prominent in the image.
            2. Try to look on the upper part of the label.
            3. Examine any barcodes in the image, as the part number might be printed above them.
            4. If there are multiple numbers, prioritize those that match the VAG part number format.
            5. Be cautious of upside-down numbers that might look like VAG numbers when flipped:
               - Check if any identified numbers make more sense when read upside-down
               - Ensure the number you're reading is oriented correctly on the label

            Remember, a valid VAG part number typically:
            - Consists of 9-11 characters
            - Is often divided into three groups (e.g., "1K2 820 015 C")
            - Has a first group of 3 characters (e.g., "1K2", "4H0")
            - Has a second group of 3 digits (e.g., "820", "907")
            - Has a third group of 3-4 characters, sometimes ending with a letter (e.g., "015 C", "801 E")

            If you find a number matching this format, please provide it.
            If you still can't find a valid number, respond with NONE.

            Response Format:
            - If a valid part number is identified: <START> [VAG Part Number] <END>
            - If no valid number is identified: <START> NONE <END>
            """
            
            self.prompt = specific_prompt

    self.reset_incorrect_predictions()  # Reset for the next page
    self.prompt = original_prompt  # Reset prompt to original
    return "NONE"
