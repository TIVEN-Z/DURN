from torch.utils import data
from os.path import join
import cv2
import torchvision.transforms as transforms

class MURAL_Loader(data.Dataset):
    """
    Dataloader MURAL
    """

    def __init__(self, root='../DATA/MURAL/', split='train', transform=False, size=320):
        self.root = root
        self.split = split
        self.transform = transform
        self.size = size

        if self.split == 'train':
            self.filelist = join(self.root, 'train.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def getpath(self):
        return self.filelist

    def __getitem__(self, index):
        if self.split == "train":
            img_lb_path = self.filelist[index].strip("\n").split(" ")
            img_path = img_lb_path[0]
            label_path = img_lb_path[1]
            imgsize = self.size
            img = cv2.imread(join(self.root, img_path), cv2.COLOR_BGR2RGB) 
            img = cv2.resize(img, (imgsize, imgsize))
            img = transforms.ToTensor()(img)
            img = img.float()

            label = cv2.imread(join(self.root, label_path),cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label,(imgsize, imgsize))
            label = transforms.ToTensor()(label)
            label = label.float()
            return img, label 





