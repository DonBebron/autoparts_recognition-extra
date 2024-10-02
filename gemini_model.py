# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep
import random

from PIL import Image
import requests

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

5. **Final Check and Marking:**
   - Ensure the number is not accompanied by a third-party manufacturer's logo (for VAG numbers).
   - The catalog number is usually placed in a prominent location on the label.

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

  def extract_number(self, response):
    return response.split('<START>')[-1].split("<END>")[0].strip()

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

    Previously incorrect predictions on this page: {', '.join(self.incorrect_predictions)}

    If the number follows these rules, respond with:
    <VALID>

    If the number does not follow these rules or seems incorrect, respond with:
    <INVALID>

    Explanation: [Brief explanation of why it's valid or invalid including the number itself]
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

      # First attempt
      answer = self.get_response(img)
      extracted_number = self.extract_number(answer)
      
      if extracted_number.upper() != "NONE":
        validation_result = self.validate_number(extracted_number)
        if "<VALID>" in validation_result:
          self.reset_incorrect_predictions()  # Reset for the next page
          return extracted_number
        else:
          print(f"First validation failed: {validation_result}")
          self.incorrect_predictions.append(extracted_number)
          
          # Second attempt with a more specific prompt
          specific_prompt = f"""
          The previously extracted number "{extracted_number}" was invalid. 
          {validation_result}
          Previously incorrect predictions on this page: {', '.join(self.incorrect_predictions)}
          Please re-examine the image carefully and try to identify a valid VAG part number.
          Remember, a valid VAG part number typically:
          - Consists of 9-11 characters
          - Is divided into three groups separated by spaces
          - Has a first group of 3 characters (e.g., "1K2", "4H0")
          - Has a second group of 3 digits (e.g., "820", "907")
          - Has a third group of 3-4 digits, sometimes followed by a letter (e.g., "015 C", "801 E")
          
          If you find a number matching this format, please provide it.
          If you still can't find a valid number, respond with NONE.

          Response Format:
          - If a valid part number is identified: <START> [VAG Part Number] <END>
          - If no valid number is identified: <START> NONE <END>
          """
          
          image_parts = [
              {
                  "mime_type": "image/jpeg",
                  "data": img.read_bytes()
              },
          ]
          prompt_parts = [image_parts[0], specific_prompt]
          
          second_answer = self.get_response(img)  # This already returns the text
          second_extracted_number = self.extract_number(second_answer)  # Remove .text here
          
          if second_extracted_number.upper() != "NONE":
            second_validation_result = self.validate_number(second_extracted_number)
            if "<VALID>" in second_validation_result:
              self.reset_incorrect_predictions()  # Reset for the next page
              return second_extracted_number
            else:
              print(f"Second validation failed: {second_validation_result}")
              self.incorrect_predictions.append(second_extracted_number)
      
      self.reset_incorrect_predictions()  # Reset for the next page
      return "NONE"
