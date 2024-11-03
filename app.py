# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# print("downloading model...")
# model_name = "facebook/blenderbot-400M-distill"
# tokenizer=AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained("./tokenizer")
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# model.save_pretrained("./model")
# print("Model downloaded and saved")

model_path = "./model"
tokenizer_path = "./tokenizer"

print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
print("Model Loaded!\n")

UTTERANCE = input("You: ")
# print("You:", UTTERANCE)
inputs = tokenizer([UTTERANCE], return_tensors="pt")
reply_ids = model.generate(**inputs)
print("Bot:", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])