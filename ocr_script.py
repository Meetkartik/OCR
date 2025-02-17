# import sys
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests
# from transformers import logging

# proxies = {
#     'http': 'http://localhost:5000',
#     'https': 'http://localhost:5000',
# }

# print('here')

# session = requests.Session()
# session.verify = False

# # Example request with SSL verification disabled
# response = requests.get('https://huggingface.co/microsoft/trocr-base-printed', verify=False)

# print(response.status_code)
# print(response.content)


# # response = requests.get('https://example.com', proxies=proxies, verify=False)
# # print(response.content,'sdsd')

# # #Disable warnings related to SSL verification issues (optional)
# # logging.set_verbosity_error()

# # Load Pretrained Models (Handwritten & Printed)
# PROCESSOR_PRINTED = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True, verify=False)
# MODEL_PRINTED = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed", use_fast=True, verify=False)

# PROCESSOR_HANDWRITTEN = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True, verify=False)
# MODEL_HANDWRITTEN = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True, verify=False)

# def extract_text(image_path, model_type="printed"):
#     """Extracts text from a cheque image using TrOCR (Handwritten or Printed)."""
    
#     image = Image.open(image_path).convert("RGB")
    
#     if model_type == "handwritten":
#         processor, model = PROCESSOR_HANDWRITTEN, MODEL_HANDWRITTEN
#     else:
#         processor, model = PROCESSOR_PRINTED, MODEL_PRINTED

#     pixel_values = processor(image, return_tensors="pt").pixel_values
#     generated_text = model.generate(pixel_values)
#     extracted_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
    
#     return extracted_text

# if __name__ == "__main__":
#     image_path = sys.argv[1]
#     model_type = sys.argv[2] if len(sys.argv) > 2 else "printed"
#     print(extract_text(image_path, model_type))



# # def download_model_with_ssl_check(model_name):
# #     print('here')
# #     url = f'https://huggingface.co/{model_name}/resolve/main/config.json'
# #     try:
# #         # Add timeout and redirect handling
# #         response = session.get(url, timeout=10, allow_redirects=True)
# #         print(f"Status Code: {response.status_code}")
# #         if response.status_code == 200:
# #             print(f"Successfully downloaded model {model_name}")
# #         else:
# #             print(f"Failed to download model {model_name} with status code {response.status_code}")
# #             print(f"Response Content: {response.text}")
# #     except requests.exceptions.Timeout:
# #         print("Request timed out.")
# #     except requests.exceptions.RequestException as e:
# #         print(f"Request failed: {e}")

# # # Test the function
# # download_model_with_ssl_check("distilbert")


import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from transformers import logging

# Define proxies for HTTP and HTTPS
# proxies = {
#     'http': 'http://localhost:8080',
#     'https': 'http://localhost:8080',
# }

# Create a custom session for handling requests
# session = requests.Session()
# session.proxies.update(proxies)
# session.verify = False  # Disable SSL verification

# Function to override the default behavior of from_pretrained to use custom session

# Use the custom session to load models
PROCESSOR_PRINTED = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True, verify=False)
MODEL_PRINTED = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed", use_fast=True, verify=False)

PROCESSOR_HANDWRITTEN = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True, verify=False)
MODEL_HANDWRITTEN = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True, verify=False)

# Function to extract text using the selected model
def extract_text(image_path, model_type="printed"):
    """Extracts text from a cheque image using TrOCR (Handwritten or Printed)."""
    image = Image.open(image_path).convert("RGB")
    
    if model_type == "handwritten":
        processor, model = PROCESSOR_HANDWRITTEN, MODEL_HANDWRITTEN
    else:
        processor, model = PROCESSOR_PRINTED, MODEL_PRINTED

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_text = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
    
    return extracted_text

# Run the extraction function when the script is executed
# if __name__ == "__main__":
#     image_path = sys.argv[1]
#     model_type = sys.argv[2] if len(sys.argv) > 2 else "printed"
#     print(extract_text(image_path, model_type))
