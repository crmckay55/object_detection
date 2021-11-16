import numpy as np
import cv2
import torch
import glob as glob
from model import create_model
import config

# set the computation device
device = torch.device(config.COMPUTATION_DEVICE)
# load the model and the trained weights
model = create_model(num_classes=config.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(
   config.MODEL_LOCATION, map_location=device
))
model.eval()

# directory where all the images are present
DIR_TEST = config.VALIDATE_SOURCE
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")


# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('\\')[-1].split('.')[0]  # Changed '/' to '\\'
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)


    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [config.CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j] + ': ' + str('{:.2%}'.format(scores[j])),        # INFO: added score to box, increased font size, formatted percent
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2, lineType=cv2.LINE_4)
        # cv2.imshow('Prediction', orig_image)
        # cv2.waitKey(1)
        cv2.imwrite(config.VALIDATE_PREDICTIONS + f"/{image_name}.jpg", orig_image, )



    print(f"Image {i + 1} done...")
    print('-' * 50)
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
