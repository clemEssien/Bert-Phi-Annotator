from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class Bert:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.nlp = None
        

    def initialize(self):
         self.tokenizer = AutoTokenizer.from_pretrained("./dslim-bert/tokenizer")
         self.model = AutoModelForTokenClassification.from_pretrained("./dslim-bert/model")
         self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def get_entities(self, sample_text, label):
        if self.nlp is None:
            self.initialize()
        ner_results = self.nlp(sample_text)
        print(ner_results)
        print(type(ner_results))

        ner_results = [result for result in ner_results if label in result['entity'] ]
        return ner_results

bert = Bert()