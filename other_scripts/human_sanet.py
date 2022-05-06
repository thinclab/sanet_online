#!/usr/bin/env python3

import random

import cv2
from detect import YOLO
from human_model_sanet import *
from time import time

import rospkg

BATCH_SIZE = 16

PRELOAD_IMAGE = False

model_path = os.path.dirname(__file__) + f'/model/sa/state_action_human/200.pth'

use_cuda = torch.cuda.is_available()
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

state_dict = {
    0: 'athome',
    1: 'onconv',
    2: 'infront',
    3: 'atbin',
    4: 'unknown'
}

oloc_dict = {      # By feeding state dict idx, we can get mdp idx
    0 : 3,      
    1 : 1,
    2 : 2,
    4 : 0,
}

eefloc_dict = {
    0 : 3,
    1 : 1,
    2 : 2,
    3 : 0,
}

pred_dict = {
    0: 'bad',
    1: 'good'
}

action_dict = {
    0: 'detect',
    1: 'pick',
    2: 'inspect',
    3: 'placeonconv',
    4: 'placeinbin',
    5: 'noop'
}

mdp_action_dict = {
    0 : 1,
    1 : 2,
    2 : 3,
    3 : 4,
    4 : 5,
    5 : 0,
}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
rospack = rospkg.RosPack()  # get an instance of RosPack with the default search paths
path = rospack.get_path('sanet_online')   # get the file path for sanet_online

class TestStateAction(object):
    def __init__(self):
        global device
        self.device = device
        self.yolo = YOLO(weightsfile = "best_realkinect.pt", conf_thres = 0.85)        
        self.model= StateActionMain().to(device)
        self.model.load_state_dict(torch.load(f'{model_path}', map_location=self.device))

    def convert_to_tensor(self):
        state_image = self.sample['state_image']        
        action_image = self.sample['action_image']

        name = self.sample['name']

        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C x H x W
        state_image = torch.from_numpy(state_image.transpose((2, 0, 1)))
        action_image = torch.from_numpy(action_image.transpose((0, 3, 1, 2)))

        tensor_sample = {'state_image': state_image,
                'action_image': action_image,
                'name': name}

        print('state_image: ', tensor_sample['state_image'])
        print('action_image: ', tensor_sample['action_image'])
        self.sample = tensor_sample

    def main(self, state_frame_list, action_frame_list):

        self.state_frame_list = state_frame_list
        self.action_frame_list = action_frame_list

        test_data = TestDataset(state_frame_list=self.state_frame_list,
                             action_frame_list = self.action_frame_list,
                             transform=transforms.Compose([
                                 RescaleTest(),
                               ToTensorTest()]))
        
        self.test_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=2,
                                     pin_memory=True, worker_init_fn=seed_worker, generator=g)

        self.model.eval()

        onion = None
        eef = None
        action = None
        detections = None
        image = self.state_frame_list[0]
        # State Detection and YOLO

        with torch.no_grad():
            for _ , sa in enumerate(self.test_loader,0):
                state_input = sa['state_image'].float().to(self.device)
                action_input = sa['action_image'].float().to(self.device)


                with torch.cuda.amp.autocast():
                    onion, eef, action = self.model(state_input,action_input)

                onion = onion/onion.sum(dim=1).unsqueeze(-1)
                eef = eef/eef.sum(dim=1).unsqueeze(-1)
                action = action/action.sum(dim=1).unsqueeze(-1)
                detections = self.yolo.detect(image)  
                predic = detections[0][-1].item()
                if predic >= 0:
                    text = f'Onion: {state_dict[torch.argmax(onion[0]).item()]}, \nEEF: {state_dict[torch.argmax(eef[0]).item()]}, \nDetections: {pred_dict[predic]}, \nAction: {action_dict[torch.argmax(action[0]).item()]}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # print("Last onion ", detections[0])
                    y0, dy = 50, 35
                    for i, line in enumerate(text.split('\n')):
                        y = y0 + i*dy
                        cv2.putText(image, line, (10,y), font, 1, (0, 255, 0), 3)
                    cv2.imwrite(path + '/data/output/' + str(round(int(time()),6)) + '.png', image)
                else: print("No Onions to be sorted!")
        return
        