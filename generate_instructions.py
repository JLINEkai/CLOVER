import openai
import datetime
import json
from retrying import retry
import time
from multiprocessing import Pool

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import pandas as pd
import os

MODEL_NAME = "gpt-3.5-turbo"


@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(20))
def get_completion(questions, img_path, index): 

    prompt = []
    openai.api_base = 'api_base XXXXXXXXXXXXXXXXX'
    openai.api_key = 'api_key XXXXXXXXXXXXXXXXXXXX'

    completion = openai.ChatCompletion.create(

        model="gpt-3.5-turbo-1106",

       messages = [{ "role" : "system",  "content" : """
                        As a specialized AI assistant focusing on pathological images, you will receive textual descriptions (caption) of figures. Please note that you do not have access to the actual images. Your task is to generate a set of question-and-answer (QA) pairs between the person inquiring about the images (user) and you as the assistant responding. The QA should be conducted as if both the user and the assistant are examining the images, without referring to textual information.
                        The following are requirements for generating question-and-answer pairs:
                            -Avoid referencing dates or magnification ratios.
                            -Focus on visual descriptions, including organizational structure, cellular morphology, potential pathological changes, location, etc.
                            -Avoid using phrases such as "mention," "title," "context," or "narrator." Instead, refer to information as being "in the image."
                            -When responding to questions, adopt an objective and responsible attitude, avoiding overconfidence, and refrain from providing medical advice or diagnostic information. Encourage users to consult healthcare professionals for more accurate advice.
                        The content should include 4-5 question-and-answer pairs related to visual aspects of the images. 
                        """ 
                    },
                    { "role": "user", "content": """Histopathological picture of younger sister showing stratified squamous lining and subepithelium shows rich collagenous fibrous connective tissue with mild inflammatory cells. (H&E). """},
                    {"role": "assistant", "content" : """question: "What type of tissue is shown in the image?"
                     answer: "The image shows stratified squamous epithelial tissue lining and subepithelial tissue, which consists of collagenous fibrous connective tissue."
                     question: "What can you say about the inflammatory cells?"
                     answer: "The image reveals the presence of mild inflammatory cells in the subepithelial region. This suggests that there might be a mild inflammatory response occurring in the tissue, which could be due to various factors such as infection, injury, or an immune reaction."
                     question: "What is the staining technique used?"
                     answer: "The staining technique used in this histopathological image is Hematoxylin and Eosin (H&E) staining. This is a commonly used technique in histology to visualize the structure and organization of tissues and cells. Hematoxylin stains the cell nuclei blue, while eosin stains the cytoplasm and extracellular matrix pink." 
                     question: "What are stratified squamous epithelium and subepithelium?"
                     answer: "the stratified squamous lining is the outermost layer composed of multiple layers of flat cells, while the subepithelium is the underlying tissue layer that supports and nourishes the epithelial cells."
                            """
                    },
                    { "role": "user", "content": questions},
        ],
        temperature=1.0,
        n=1,
        top_p=1.0,
    )
    print(index + " " + questions + "\n")
    print("image: " + img_path + '\n')
    print(completion.choices[0].message.content)
    print('#############################')
    return '\n'.join(questions), completion.choices[0].message.content

if __name__=="__main__":
    pool = Pool(processes=6)
    result = []
    
    #quilt_1m csv path
    df = pd.read_csv("./merge_quilt_filte.csv")   
    df = df[30000:35000]

    
    for i in range(len(df)):

        item = df.iloc[i]
        if type(item['caption']) == float:
            continue

        result.append([pool.apply_async(get_completion, args=(item['caption'], item['image_path'], str(i))), item['image_path'], item['caption']])
        
 
    pool.close()
    pool.join()

    
    json_data = [{"QA": item[0].get()[1], "path": item[1], 'caption': item[2]} for item in result]
    
    
    with open('./quilt_filte_30000_35000.json', 'w') as f:
        # for item in result:
        try:
            json.dump(json_data, f, indent=4)
            f.write('\n')
        except:
            print('Not Save')