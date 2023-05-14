import json
from datasets import load_dataset
from speech_collator import SpeechCollator, create_speaker2idx, create_phone2idx

dataset = load_dataset("cdminix/libritts-aligned", split="train")

# Create speaker2idx and phone2idx
speaker2idx = create_speaker2idx(dataset)
phone2idx = create_phone2idx(dataset)

# save to json
with open("speaker2idx.json", "w") as f:
    json.dump(speaker2idx, f)
with open("phone2idx.json", "w") as f:
    json.dump(phone2idx, f)

print("speaker2idx:", speaker2idx)
print("phone2idx:", phone2idx)
