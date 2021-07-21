from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def get_entities(embedding, model_name, sample_text):

    tokenizer = AutoTokenizer.from_pretrained(embedding)   
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    ner_results = nlp(sample_text)

    return ner_results

# import os

# files = os.listdir("../ms-bert")
# print(files)