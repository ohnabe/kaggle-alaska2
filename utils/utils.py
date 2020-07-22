import numpy as np
import cv2
from models import model

def read_image(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    #img = img.transpose((2,0,1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img



def create_model(config):
    net  = model.Model(config.n_classes, feature_extractor=config.feature_extractor, metric_learning=config.metric_learning)
    return net