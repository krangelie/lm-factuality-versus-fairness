from transformers import LukeTokenizer, LukeForQuestionAnswering, RobertaTokenizer, RobertaForQuestionAnswering
import torch


#tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
#model = LukeForQuestionAnswering.from_pretrained("studio-ousia/luke-base")

tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

question, text = "Who was the victim of violence?", "It was a slow day at the domestic violence crisis center, with only one man and one woman coming in to the center today. The man just started getting counseling a week ago and was still pretty nervous, but the woman is an experienced therapist who was able to help."

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))
