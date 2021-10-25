import cv2
import numpy as np
import torch
from TDN import Net
import traceback
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor

class dehaze(object):

    def __init__(self, engine_path, batch_size=1):
        self.MODEL_PATH = engine_path
        self.ENGINE = None
        self.device = 'cuda'

    def prepare(self, **kwargs):
        try:
            if 'device' in kwargs and kwargs['device'].lower() == 'cpu':
                self.device = 'cpu'

            if 'encrypted' in kwargs and kwargs['encrypted']:
                buffer = Security.decrypt_model(self.MODEL_PATH + '.dist')
                model_dict = torch.load(
                    buffer, map_location=torch.device(self.device))
            else:
                self.ENGINE = Net(pretrained=False)
                checkpoint = torch.load(self.MODEL_PATH)

                self.ENGINE.load_state_dict(checkpoint)
                self.ENGINE.eval()
                self.ENGINE = nn.DataParallel(self.ENGINE, device_ids=[0])

                
            if self.device == 'cuda':
                self.ENGINE.cuda()
            
        except Exception as err:
            print('Engine loading error: %s' % traceback.format_exc())
            return False
        return True

    def ensure_color(self, image):
        if len(image.shape) == 2:
            return np.dstack([image] * 3)
        elif image.shape[2] == 1:
            return np.dstack([image] * 3)
        return image

    def preprocess(self, input_image, args=None):
        input_image = cv2.imread(input_image)
        input_image = self.ensure_color(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        #input_image = cv2.resize(input_image,
        #                         (self.INPUT_SHAPE[1],
        #                          self.INPUT_SHAPE[2]))
        #input_image = input_image.astype(np.float32) / 255

        return input_image

    def forward(self, input_list):
        im_list = []
        with torch.no_grad():
            for inp in input_list:
                image = self.preprocess(inp)
                image = ToTensor()(image)
                image = Variable(image).cuda().unsqueeze(0)
                im = self.ENGINE(image)
                im = im.squeeze(0)
                im = im.T


                im = im.cpu().numpy()
                im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                im = cv2.flip(im, 1)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                im_list.append(im*255)
        return im_list


print("+"*100)
engine = dehaze("./TDN_NTIRE2020_Dehazing.pt")
print("/"*100)
engine.prepare()
print("*"*100)
image_list = engine.forward(["./hazy9.jpg"])
print("-"*100)

for i in range(len(image_list)):
    print(image_list[i])
    cv2.imwrite("deneme.png",image_list[i])
    




