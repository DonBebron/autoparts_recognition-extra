# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep
import random
from collections import Counter

from PIL import Image
import requests
import re

DEFAULT_PROMPT = """identify Main Catalog Number from photo by this Algorithm

**Note:** This algorithm is designed to identify and verify VAG part numbers. Assume all part numbers belong to VAG by default.

1. **Identify Potential Numbers on the Photo:**
   - Examine all numbers and markings in the image.
   - Look for numbers that resemble catalog numbers, which typically include combinations of digits and letters.

2. **Analyze the Structure of the Numbers:**
   - VAG Catalog Numbers (Part Numbers) typically consist of 10-11 characters.
   - They are often, but not always, divided into visible groups.
   - The structure usually follows this pattern:
     - First part: 3 characters (e.g., "5Q0", "8S0")
     - Middle part: 3 digits (e.g., "937", "907")
     - Last part: 3-4 characters, which may include digits and/or letters (e.g., "085B", "468D")

3. **Determine the Brand by the Numbers:**
   - VAG Numbers: Numbers that closely follow the structure described above are likely VAG numbers.
   - OEM Numbers: If the number is accompanied by a third-party manufacturer's logo and doesn't follow the VAG format, it's likely an OEM number.

4. **Verify the Accuracy of the Number:**
   - Match with Known Formats: Compare the identified number with the VAG format described above.
   - Check for Consistency: Ensure the number maintains a consistent structure throughout.
   - Pay special attention to the last part of the number:
     - It should not contain any digits after known letter suffixes (e.g., "AD" should not be followed by digits)
     - If it ends with a single letter, make sure it's not missing (e.g., "T" at the end)

5. **Final Check and Marking:**
   - Ensure the number is not accompanied by a third-party manufacturer's logo (for VAG numbers).
   - The catalog number is usually placed in a prominent location on the label.
   - Double-check that you haven't included any extra digits or characters that don't belong to the actual part number.

6. **Avoid Previously Incorrect Predictions:**
   - Do not return any numbers that have been previously identified as incorrect.
   - If you find a number that matches the structure but has been marked as incorrect before, look for alternative numbers in the image.

Please follow the above steps to recognize the correct detail number and format the response as follows:

**Response Format:**
- If a valid part number is identified: `<START> [VAG Part Number] <END>`
- If no valid number is identified or all potential numbers have been previously marked as incorrect: `<START> NONE <END>`
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
    self.incorrect_predictions = Counter()
    self.max_attempts_per_number = 3
    self.max_total_attempts = 5

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

  def get_response(self, img, prompt):
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": img.read_bytes()
        },
    ]
    prompt_parts = [
        image_parts[0],
        prompt,
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

    Rules for validation:
    1. The number should consist of 10-11 characters.
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

    Previously incorrect predictions on this page: {', '.join(self.incorrect_predictions.keys())}

    If the number follows these rules, respond with:
    <VALID>

    If the number does not follow these rules or seems incorrect, respond with:
    <INVALID>

    Explanation: [Brief explanation of why it's valid or invalid including the number itself]
    """

    response = self.validator_model.generate_content(prompt)
    return response.text

  def double_check_confused_digits(self, extracted_number):
    prompt = f"""
    Double-check the following VAG part number for commonly confused digits and potential errors: {extracted_number}

    Focus on:
    1. '9' vs '8': Ensure these are correctly identified.
    2. '0' vs 'O': Confirm all instances are digits, not letters.
    3. '1' vs 'I': Verify these are correctly distinguished.
    4. Check the last part of the number:
       - It should not contain any digits after known letter suffixes (e.g., "AD" should not be followed by digits)
       - If it ends with a single letter, make sure it's not missing (e.g., "T" at the end)
    5. Ensure no extra digits or characters are included that don't belong to the actual part number.

    If you believe any digits might be incorrect or if there are any other issues, suggest the most likely correct version.

    Response format:
    <CORRECTED> [Corrected number if changes are needed]
    or
    <UNCHANGED> [Original number if no changes are needed]

    Explanation: [Brief explanation of any changes or why no changes were needed]
    """

    response = self.validator_model.generate_content(prompt)
    return response.text

  def reset_incorrect_predictions(self):
    self.incorrect_predictions.clear()

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

    total_attempts = 0
    none_count = 0
    max_none_attempts = 3  # Maximum number of consecutive "NONE" responses before giving up

    while total_attempts < self.max_total_attempts:
        prompt_with_incorrect = f"{self.prompt}\n\nPreviously incorrect predictions: {', '.join(self.incorrect_predictions.keys())}"

        answer = self.get_response(img, prompt_with_incorrect)
        extracted_number = self.extract_number(answer)

        if extracted_number.upper() == "NONE":
            none_count += 1
            if none_count >= max_none_attempts:
                print(f"No valid numbers found after {none_count} consecutive attempts. Stopping.")
                break
            total_attempts += 1
            continue

        none_count = 0  # Reset the none_count if we get a non-NONE response

        if self.incorrect_predictions[extracted_number] >= self.max_attempts_per_number:
            print(f"Skipping previously incorrect prediction: {extracted_number}")
            total_attempts += 1
            continue

        validation_result = self.validate_number(extracted_number)
        if "<VALID>" in validation_result:
            double_check_result = self.double_check_confused_digits(extracted_number)
            if "<CORRECTED>" in double_check_result:
                corrected_number = double_check_result.split("<CORRECTED>")[1].split("\n")[0].strip()
                print(f"Number corrected after double-check: {corrected_number}")
                self.reset_incorrect_predictions()
                return corrected_number
            elif "<UNCHANGED>" in double_check_result:
                self.reset_incorrect_predictions()
                return extracted_number
        else:
            print(f"Validation failed: {validation_result}")
            self.incorrect_predictions[extracted_number] += 1
            total_attempts += 1

    self.reset_incorrect_predictions()  # Reset for the next image
    return "NONE"
