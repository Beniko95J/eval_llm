import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:1"

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

processor = AutoProcessor.from_pretrained('HuggingFaceM4/idefics2-8b-base')
model = AutoModelForVision2Seq.from_pretrained('HuggingFaceM4/idefics2-8b-base', torch_dtype=torch.float16).to(DEVICE)

# Create inputs
prompts = [
    "<image>In this image, we can see the city of New York, and more specifically the Statue of Liberty.<image>In this image,",
    "In which city is that bridge located?<image>",
]
images = [[image1, image2], [image3]]
# import pdb; pdb.set_trace()
# images = [[image3]]
inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# Generate
with torch.cuda.amp.autocast(dtype=torch.float16):
    generated_ids = model.generate(**inputs, max_new_tokens=500)

generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
import pdb; pdb.set_trace()

print(generated_texts)
