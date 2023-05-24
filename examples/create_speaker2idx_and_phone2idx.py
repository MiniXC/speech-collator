import json
from torch.utils.data import DataLoader
from datasets import load_dataset
from speech_collator import SpeechCollator, create_speaker2idx, create_phone2idx
from speech_collator.measures import PitchMeasure, EnergyMeasure, VoiceActivityMeasure

dataset = load_dataset("cdminix/libritts-aligned", split="train")

# # Create speaker2idx and phone2idx
# speaker2idx = create_speaker2idx(dataset)
# phone2idx = create_phone2idx(dataset)

# # save to json
# with open("data/speaker2idx.json", "w") as f:
#     json.dump(speaker2idx, f)
# with open("data/phone2idx.json", "w") as f:
#     json.dump(phone2idx, f)

speaker2idx = json.load(open("data/speaker2idx.json"))
phone2idx = json.load(open("data/phone2idx.json"))

# Create SpeechCollator
speech_collator = SpeechCollator(
    speaker2idx=speaker2idx,
    phone2idx=phone2idx,
    measures=[PitchMeasure(), EnergyMeasure(), VoiceActivityMeasure()],
    return_keys=["measures"]
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=speech_collator.collate_fn,
)

for batch in dataloader:
    print(batch["measures"]["voice_activity"][0])
    break