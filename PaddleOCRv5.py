import os
os.environ.setdefault("PADDLE_DISABLE_ONE_DNN", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
    enable_mkldnn=False,
)

script_dir = os.path.dirname(__file__)
image_dir = os.path.join(script_dir, "test images")
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"Image folder not found: {image_dir}")

selected_filenames = ["image 5.png"]  # Add filenames here

if not selected_filenames:
    raise ValueError("No images selected.")

image_paths = []
for filename in selected_filenames:
    path = os.path.join(image_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    image_paths.append(path)

output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

for image_path in image_paths:
    print(f"Processing: {image_path}")
    result = ocr.predict(input=image_path)
    name = os.path.splitext(os.path.basename(image_path))[0]
    output_prefix = os.path.join(output_dir, name)

    for res in result:
        res.print()
        res.save_to_img(output_prefix)
        res.save_to_json(output_prefix)