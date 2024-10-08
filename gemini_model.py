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
    Perform a rigorous validation of the following VAG (Volkswagen Audi Group) part number: {extracted_number}

    Comprehensive Validation Criteria:
    1. Length and Format:
       - Total length should be 11-13 characters (including spaces).
       - Format: [First Number] [Middle Number] [Final Number] [Index] [Software Variant]
       - Example: 5K0 937 087 AC Z15

    2. Detailed Component Analysis:
       a) First Number (3 characters):
          - First two digits: Must match known vehicle types (e.g., 3D, 1J, 8L)
          - Third digit: Should indicate a valid body shape or variant (0-9)
       b) Middle Number (3 digits):
          - First digit: Must correspond to a valid main group (1-9)
          - Last two digits: Should be a valid subgroup within the main group
       c) Final Number (3 digits):
          - Should be a logical part identifier within the subgroup
          - Check if odd/even aligns with left/right part designation (if applicable)
       d) Index (1-2 letters, if present):
          - Should be valid alphabetic characters
          - 'X' might indicate an exchange or remanufactured part
       e) Software Variant (2-3 characters, if present):
          - Often starts with 'Z' followed by numbers

    3. Logical Consistency:
       - Ensure the combination of vehicle type, main group, and subgroup is plausible.
       - Check if the part number aligns with any visible part characteristics in the image.

    4. Special Cases and Exceptions:
       - Standard parts: May start with 9xx.xxx or 052.xxx
       - Color codes: Look for known color designations (e.g., GRU for primed parts)

    5. Character Verification:
       - Double-check for common misinterpretations:
         '1' vs 'I', '0' vs 'O', '8' vs 'B', '5' vs 'S', '2' vs 'Z'
       - Ensure no inappropriate characters are present (e.g., symbols, punctuation)

    6. Contextual Validation:
       - Consider if the number makes sense for the apparent part or vehicle type visible in the image.
       - Check for any contradictory information in the image (e.g., non-VAG logos)

    7. Upside-Down Check:
       - Verify the number isn't an inverted non-VAG number that coincidentally looks valid.

    Previously incorrect predictions on this page: {', '.join(self.incorrect_predictions)}

    Based on this comprehensive analysis, respond with:
    <VALID> if the number passes all relevant checks.
    <INVALID> if it fails any crucial validation criteria.

    Explanation: [Provide a detailed explanation of your validation process, highlighting any specific points of concern or confirmation]
    """

    response = self.validator_model.generate_content(prompt)
    return response.text

  def double_check_confused_digits(self, extracted_number):
    prompt = f"""
    Perform an in-depth analysis of the following VAG part number, focusing on potential digit confusions and errors: {extracted_number}

    Detailed Examination Points:
    1. Digit Pair Analysis:
       - '9' vs '8': Carefully distinguish, considering context (e.g., is it more likely to be a subgroup 9xx or 8xx?)
       - '0' vs 'O': Ensure all instances are digits, not letters. Check against known formats.
       - '1' vs 'I': Verify correct identification, especially in the first number group.
       - '5' vs 'S': Confirm in context, particularly in the index portion.
       - '2' vs 'Z': Differentiate, especially in the software variant section.

    2. Structural Integrity Check:
       a) First Number (3 characters):
          - Verify the first two digits match a known vehicle type.
          - Ensure the third digit is a valid body shape indicator (0-9).
       b) Middle Number (3 digits):
          - Confirm the first digit corresponds to a valid main group (1-9).
          - Check if the last two digits form a valid subgroup for that main group.
       c) Final Number (3 digits):
          - Assess if it's a plausible part identifier within the subgroup.
          - For paired parts, verify if odd/even aligns with left/right designation.

    3. Index and Software Variant Verification:
       - If present, ensure the index uses valid characters (typically 1-2 letters).
       - For software variants, confirm it starts with 'Z' if applicable.

    4. Logical Sequence Analysis:
       - Evaluate if the entire number forms a logical sequence for a VAG part.
       - Check for any digits that seem out of place or inconsistent with the rest of the number.

    5. Special Format Considerations:
       - For numbers starting with 9xx.xxx or 052.xxx, verify they follow standard part numbering rules.
       - For potential color codes or exchange parts, ensure proper formatting (e.g., 'X' for exchange parts).

    6. Contextual Plausibility:
       - Consider if the number makes sense for the type of part visible in the image.
       - Check for any contradictions between the number and visible part characteristics.

    Based on this comprehensive analysis, respond with:
    <CORRECTED> [Corrected number] if any changes are needed, along with a detailed explanation of each correction.
    <UNCHANGED> [Original number] if no changes are needed, with an explanation of why it's deemed correct.

    Explanation: [Provide a thorough breakdown of your analysis, including any potential issues identified and how you resolved them]
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
                # Add an extra check for commonly confused digits
                double_check_result = self.double_check_confused_digits(extracted_number)
                if "<CORRECTED>" in double_check_result:
                    corrected_number = double_check_result.split("<CORRECTED>")[1].split("\n")[0].strip()
                    print(f"Number corrected after double-check: {corrected_number}")
                    # Validate the corrected number
                    final_validation = self.validate_number(corrected_number)
                    if "<VALID>" in final_validation:
                        self.reset_incorrect_predictions()
                        self.prompt = original_prompt  # Reset prompt to original
                        return corrected_number
                    else:
                        print(f"Corrected number failed validation: {final_validation}")
                        self.incorrect_predictions.append(corrected_number)
                elif "<UNCHANGED>" in double_check_result:
                    self.reset_incorrect_predictions()
                    self.prompt = original_prompt  # Reset prompt to original
                    return extracted_number
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
            2. Try to look on the upper part of the label or above any barcodes.
            3. Remember the VAG part number structure: [First Number] [Middle Number] [Final Number] [Index] [Software Variant]
               Example: 5K0 937 087 AC Z15
            4. The first three parts (First Number, Middle Number, Final Number) are most crucial and should always be present.
            5. The Index and Software Variant may not always be visible or present.
            6. Be cautious of upside-down numbers that might look like VAG numbers when flipped.

            Remember, a valid VAG part number typically:
            - Consists of 11-13 characters (including spaces)
            - Has a first group of 3 characters (e.g., "5K0", "4H0")
            - Has a second group of 3 digits (e.g., "937", "907")
            - Has a third group of 3 digits (e.g., "087", "085")
            - May have an index of 1-2 letters (e.g., "AC", "B")
            - May have a software variant, often starting with Z (e.g., "Z15", "Z4")

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
