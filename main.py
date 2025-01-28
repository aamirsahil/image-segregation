from utils import getFilePaths, loadModel
from PIL import Image
import math
import os

def main():
    # load files
    input_path = "./input/test/*"
    file_paths = getFilePaths(input_path)
    output_path = "./ouput/"
    model_path = "./model/"
    model_url = 'dandelin/vilt-b32-finetuned-vqa'

    processor, model = loadModel(model_path, model_url)

    # process each file
    for file_path in file_paths[:5]:
        filename = file_path.split('\\')[-1]
        image = Image.open(file_path)
        
        # calculate final score
        score = 3
        # if final_score > 3.5:
        #     final_score = math.ceil(final_score)
        # elif final_score < 2.5:
        #     final_score = math.floor(final_score)
        # else:
        #     final_score = 3

        # Define a question
        question = "is the picture blurry, answer yes or no?"

        # Preprocess the image and question
        encoding = processor(image, question, return_tensors="pt")

        # Run inference
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_answer_idx = logits.argmax(-1).item()

        # Get the predicted answer
        answer = model.config.id2label[predicted_answer_idx]
        
        print(f"Image Id: {filename.split('_')[0]}")
        print(f"Image Label: {filename.split('_')[-1].split['.'][0]}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        # save to each file
        if not os.path.exists(output_path + f"{score}"):
            os.mkdir(output_path + f"{score}")

        image.save(output_path + f"{score}/" + filename)

if __name__=="__main__":
    main()