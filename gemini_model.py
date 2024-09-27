# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep

from PIL import Image
import requests

class GeminiInference():
  def __init__(self, api_key,  model_name = 'gemini-1.5-pro', prompt=None):
    self.gemini_key = api_key
    self.prompt = prompt

    genai.configure(api_key=self.gemini_key)
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
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


  def get_response(self, img):
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": img.read_bytes()
        },
    ]
    prompt_parts = [
        image_parts[0],
        ("""identify Main Catalog Number from photo by this Algorithm


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
""" if self.prompt==None else self.prompt),
    ]
    response = self.model.generate_content(prompt_parts)
    return response.text

  def extract_number(self, response):
    return response.split('<START>')[-1].split("<END>")[0] 

  def __call__(self, image_path):

      # Validate that an image is present
      if image_path.startswith('http'):
        # read remote img bites
        img = Image.open(requests.get(image_path, stream=True).raw)
        # save image to local "example_image.jpg"
        img.save("example_image.jpg")
        image_path = "example_image.jpg"

      if not (img := Path(image_path)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

      answer = self.get_response(img)

      sleep(5)
      return self.extract_number(answer)

