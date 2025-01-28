from transformers import ViltProcessor, ViltForQuestionAnswering
import os

def loadModel(model_path, model_url):
    model_name = model_url.split('/')[-1]

    if not os.path.exists(model_path + model_name):
        processor = ViltProcessor.from_pretrained(model_url)
        model = ViltForQuestionAnswering.from_pretrained(model_url)

        processor.save_pretrained(model_path + model_name)
        model.save_pretrained(model_path + model_name)
    else:
        processor = ViltProcessor.from_pretrained(model_path + model_name)
        model = ViltForQuestionAnswering.from_pretrained(model_path+ model_name)

    return processor, model