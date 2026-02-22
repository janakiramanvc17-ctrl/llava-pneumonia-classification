import torch
import medmnist
from medmnist import INFO
from torchvision import transforms
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ------------------------------
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------
# Load Dataset
# ------------------------------
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

test_dataset = DataClass(split='test', download=True)

# ------------------------------
# Image Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3)
])

# ------------------------------
# Load LLaVA Model
# ------------------------------
model_id = "llava-hf/llava-1.5-7b-hf"

processor = LlavaProcessor.from_pretrained(model_id)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# ------------------------------
# Helper Function
# ------------------------------
def extract_prediction(response):
    response = response.lower()

    if "pneumonia" in response:
        return 1
    elif "normal" in response:
        return 0
    else:
        return -1

# ------------------------------
# Evaluate on Multiple Samples
# ------------------------------
all_preds = []
all_labels = []

total_samples = 100  # change if needed

for i in range(total_samples):
    image, label = test_dataset[i]
    image = transform(image)

    prompt = (
        "USER: <image>\n"
        "Is this chest X-ray Normal or Pneumonia? "
        "Answer only Normal or Pneumonia.\nASSISTANT:"
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    response = processor.decode(output[0], skip_special_tokens=True)

    pred = extract_prediction(response)

    if pred != -1:
        all_preds.append(pred)
        all_labels.append(label)

# ------------------------------
# Metrics
# ------------------------------
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

print("Accuracy:", accuracy)
print(classification_report(all_labels, all_preds))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Save results
with open("results/accuracy_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(str(classification_report(all_labels, all_preds)))
