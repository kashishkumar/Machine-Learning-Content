from transformers import pipeline

classifier = pipeline("feature-extraction")

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))

feature_extractor = pipeline("feature-extraction")
vectors = feature_extractor(
    ["I don't like this movie.", "I am amazed at the confidence of my Prime Minister."]
)
print(len(vectors[0][0][0]))  # 768

query_engine = pipeline("question-answering")
print(query_engine({"question": "What is my name?", "context": "My name is Sylvain."}))
# Sylvain

summarizer = pipeline("summarization")
print(
    summarizer(
        "Hugging Face is a French company based in New-York. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window. It focuses on natural language processing, in particular the transformers architecture.",
        max_length=50, min_length=5
    )
)

NER_engine = pipeline("ner", grouped_entities=True)
print(NER_engine("My name is Sylvain and I work at Hugging Face in Brooklyn."))

zero_shot_classifier = pipeline("zero-shot-classification")
print(
    zero_shot_classifier(
        "This is an amazing day to be alive!",
        candidate_labels=["life", "politics", "business"],
    )
)

text_generator = pipeline("text-generation")
print(text_generator("As far as I am concerned, I will", max_length=30, do_sample=False))

