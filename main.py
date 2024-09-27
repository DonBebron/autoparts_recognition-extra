from config import * 
from picker_model import TargetModel
from gemini_model import GeminiInference
from collect_data import collect_links, encode_images

import argparse
from tqdm.auto import tqdm

import telebot
import numpy as np
import pandas as pd  
import pickle 
import requests

from io import BytesIO
from PIL import Image
from IPython.display import clear_output

cfg = Config
logs = Logs()

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running the Extra")
    
    parser.add_argument('--model', type=str, required=True, help="The name of the model to use, e.g., 'gemini'")
    parser.add_argument('--gemini-api', type=str, required=True, help="API key for the Gemini model")
    parser.add_argument('--gemini-api-model', type=str, default='gemini-1.5-pro', required=False, help="Gemini model u going to use")
    parser.add_argument('--prompt', type=str, default=None, required=False, help="source to txt file write prompt written inside")
    parser.add_argument('--first-page-link', type=str, default='https://injapan.ru/category/2084017018/currency-USD/mode-1/condition-used/page-1/sort-enddate/order-ascending.html', required=False, help="")
    parser.add_argument('--save-file-name', type=str, default='recognized_data', required=False, help="")
    parser.add_argument('--ignore-error', action='store_true', help="Ignore errors and continue processing")
    parser.add_argument('--max-steps', type=int, default=3, required=False, help="Maximum steps to collect links")
    parser.add_argument('--max-links', type=int, default=90, required=False, help="Maximum number of links to collect")

    args = parser.parse_args()
    
    if (args.prompt) == None: 
        prompt = None
    else: 
        try: 
            print(f"Reading file in {args.prompt}")
            with open(args.prompt, 'r') as f: 
                prompt = f.read() 
                print("Readed Prompt: ", prompt)
        except Exception as e: 
            print(f"Error while reading the {args.prompt}:", e)
            prompt = None 

    return (
        args.model, args.gemini_api, 
        {
            'gemini_model': args.gemini_api_model, 
            'prompt': prompt, 
            'main_link': args.first_page_link, 
            'savename': args.save_file_name,
            'ignore_error': args.ignore_error,
            'max_steps': args.max_steps,
            'max_links': args.max_links
         },)

def encode(link:str, 
           picker:TargetModel, 
           model:GeminiInference) -> dict: 
    page_img_links = picker.processor.parse_images_from_page(link)
    page_img_links = list(set(page_img_links))
    
    images_probs = picker.do_inference_return_probs(page_img_links)
    for target_image_link, score in [(i['image_link'], i['score']) for i in images_probs]: 
        print(f'Trying to predict on image {target_image_link} with score {score}')
        detail_number = str(model(target_image_link))
                
        if detail_number.lower().strip() == 'none'.lower(): 
            print("Detail number not found, trying again...") 
            continue
        else: 
            break
    
    if detail_number.lower().strip() == 'none'.lower(): 
        print("Trying again...")
        
        for target_image_link, score in [(i['image_link'], i['score']) for i in images_probs]: 
            print(f'Trying to predict on image {target_image_link} with score {score}')
            detail_number = str(model(target_image_link))
                    
            if detail_number.lower().strip() == 'none'.lower(): 
                print("Detail number not found, trying again...") 
                continue
            else: 
                break
        
    print("Predicted number id:", detail_number)

    parsed_info = picker.processor.load_product_info(link)
    return {"predicted_number": detail_number, 
            "url": link, 
            "price": parsed_info['price'], 
            "correct_image_link": target_image_link, 
            "incorrect_image_links": ", ".join([l for l in page_img_links if l != target_image_link])}

def save_intermediate_results(result, filename):
    try:
        pd.DataFrame(result).to_excel(f"{filename}.xlsx", index=False)
        print(f"Intermediate results saved to {filename}.xlsx")
    except Exception as e:
        print(f"Error saving intermediate results to Excel: {e}. Saving in pickle format instead.")
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(result, f)

def reduce(main_link:str, 
           picker:TargetModel, 
           ignore_error:bool = False, 
           max_steps:int = 3, 
           max_links:int = 90, 
           savename:str = 'recognized_data', 
           **kwargs): 
    all_links = collect_links(picker, main_link, max_pages=max_steps, max_links=max_links)
    all_links = list(set(all_links))
               
    result = {"predicted_number": list(), 
              "url": list(), 
              "price": list(), 
              "correct_image_link": list(), 
              "incorrect_image_links": list()}
    
    for i, page_link in tqdm(enumerate(all_links), total=len(all_links)):     
        try: 
            print(f"Processing {i+1}/{len(all_links)} link")
            for (k, v) in encode(page_link,picker, **kwargs).items(): 
                result[k].append(v)

            if (i + 1) % 10 == 0:  # Save every 10 iterations
                save_intermediate_results(result, f"{savename}_part_{i // 10 + 1}")

            clear_output(wait=False)
        except Exception as e: 
            print(e)
            if ignore_error:
                continue
            else:
                break

    return result

if __name__ == "__main__": 
    # Parse important variables
    model_name, gemini_api, addictional_data = parse_args() 

    # Initalize models
    assert model_name in ['gemini'], "There no available model you lookin for"

    if model_name == 'gemini': 
        model = GeminiInference(api_key =gemini_api, model_name =addictional_data['gemini_model'], prompt=addictional_data['prompt'])
    else: 
        model = None 

    picker = TargetModel()

    encoding_result = reduce(
        addictional_data['main_link'], 
        picker = picker, 
        model=model,
        ignore_error=addictional_data['ignore_error'],
        max_steps=addictional_data['max_steps'],
        max_links=addictional_data['max_links'],
        savename=addictional_data['savename']
    )

    # Save final results
    try:
        pd.DataFrame(encoding_result).to_excel(f"{addictional_data['savename']}.xlsx", index=False)
    except Exception as e:
        print(f"Error saving to Excel: {e}. Saving in pickle format instead.")
        with open(f'{addictional_data["savename"]}.pkl', 'wb') as f:
            pickle.dump(encoding_result, f)
