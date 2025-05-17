import torch
from PIL import Image

# from transformers import TableTransformerForObjectDetection
# model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

# Load model directly
import torch
from PIL import Image
import os
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def main():
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")

    print(f'processor loaded: {processor}')
    print(f'model loaded: {model}')
    print(f'model.config: {model.config}')
    print(f'model.config.id2label: {model.config.id2label}')
    # print(f'model weights: {model.state_dict()}')
    print(f'number of model weights: {sum(p.numel() for p in model.state_dict().values())}')



    # from transformers import AutoImageProcessor, AutoModelForObjectDetection

    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")

    input_folder = 'example_data/images'
    images = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.jpg')]
    print(f'images: {images}')

    for image in images:
        process_image(image, processor, model)


def process_image(path: str, processor, model):
    # Load an image (replace 'path_to_image.jpg' with your image path)
    image = Image.open(path)

    # Preprocess the image
    inputs = processor(image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process outputs (e.g., extracting bounding boxes and labels)
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes

    # You can further process logits and pred_boxes as needed
    print(logits)
    print(pred_boxes)

if __name__ == "__main__":
    main()


# def cell_detection(file_path):

#     image = Image.open(file_path).convert("RGB")
#     print()
#     width, height = image.size
#     image.resize((int(width*0.5), int(height*0.5)))


#     encoding = feature_extractor(image, return_tensors="pt")
#     encoding.keys()

#     with torch.no_grad():
#         outputs = model(**encoding)


#     target_sizes = [image.size[::-1]]
#     results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
#     plot_results(image, results['scores'], results['labels'], results['boxes'])
#     model.config.id2label

# file_path = r"C:\Users\blur\Coding\blur\Segmentation\test_images\perfect_table_inkltext.png"
# #file_path = r"C:\Users\blur\Coding\blur\Segmentation\test_images\perfect_table.png"
# #file_path = r"C:\Users\blur\Coding\blur\Segmentation\test_images\test_kaggle.png"
# cell_detection(file_path)