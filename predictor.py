from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load
import torch
import sys
import cv2
from PIL import Image
import torchvision.transforms as T
sys.path.insert(0, './yolov5')


# Load the YOLOv5 model
model = attempt_load('best.pt').to(torch.device('cpu'))
stride = int(model.stride.max())  # calculate output stride


def detectAndDraw(path):
    imgWRects = cv2.imread(path)
    he, wi, _ = imgWRects.shape
    # Preprocess the image
    imgWRects = cv2.cvtColor(imgWRects, cv2.COLOR_BGR2RGB)
    imgWRects = Image.fromarray(imgWRects)
    transform = T.Compose([T.Resize((416, 416)), T.ToTensor()])
    imgWRects = transform(imgWRects).unsqueeze(0)

    # Make predictions on the image
    with torch.no_grad():
        pred = model(imgWRects.float(), augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.5)
    # Load image
    imgWRects = cv2.imread(path)
    xFactor = wi/416
    yFactor = he/416
    # Print the bounding boxes
    for det in pred[0]:
        if det is not None:
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1 *= xFactor
            x2 *= xFactor
            y1 *= yFactor
            y2 *= yFactor
            tu1 = (int(x1), int(y1))
            tu2 = (int(x2), int(y2))
            # Draw rectangle
            cv2.rectangle(imgWRects, pt1=tu1, pt2=tu2,
                          color=(0, 0, 255), thickness=4)
            cv2.putText(imgWRects, f'conf: {int(conf*100)}%', (
                tu1[0], tu1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), thickness=2)

    return imgWRects


def test(paths):

    for tstPath in paths:
        # Display image
        cv2.namedWindow(tstPath, cv2.WINDOW_NORMAL)
        cv2.imshow(tstPath, detectAndDraw(f'car_ims/cars_test/{tstPath}'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# imgToShow=cv2.rectangle(cv2.imread(path),pt1=tu1,pt2=tu2,color=)


test([
    '07638.jpg', '07671.jpg', '07804.jpg', '07824.jpg'
])
