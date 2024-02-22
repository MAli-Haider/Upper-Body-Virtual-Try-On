import os
from PIL import Image
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from networks.u2net import U2NET
device = 'cuda'

class ImageProcessor:
    """Image processing class for cloth segmentation using U2NET."""

    def __init__(self, checkpoint_path='cloth_segm_u2net_latest.pth'):
        self.net = U2NET(in_ch=3, out_ch=4)
        self.net = self.load_checkpoint_mgpu(self.net, checkpoint_path)
        self.net = self.net.to(device)
        self.net = self.net.eval()

        # Palette for visualization
        self.palette = self.get_palette(4)

    def load_checkpoint_mgpu(self, model, checkpoint_path):
        """Load checkpoint for the model."""
        if not os.path.exists(checkpoint_path):
            print("----No checkpoints at the given path----")
            return model

        model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        new_state_dict = OrderedDict()

        for k, v in model_state_dict.items():
            name = k[7:]  # Remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("----Checkpoints loaded from path: {}----".format(checkpoint_path))
        return model

    def get_palette(self, num_cls):
        """Get the color map for visualizing the segmentation mask."""
        palette = [0] * (num_cls * 3)

        for j in range(0, num_cls):
            lab = j
            palette[j * 3: (j + 1) * 3] = [255] * 3
            i = 0

            while lab:
                palette[j * 3: (j + 1) * 3] = [255] * 3
                i += 1
                lab >>= 3

        return palette

    def process_image(self, image_path, result_dir='/content/inputs/test/cloth-mask'):
        """Process an image for cloth segmentation."""
        img = Image.open(image_path).convert('RGB')
        img_size = img.size

        # Resize image for processing
        img = img.resize((768, 768), Image.BICUBIC)
        image_tensor = self.transform_image(img)

        # Process image through the U2NET model
        output_tensor = self.net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

        # Convert the output to an image and resize it to the original size
        output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
        output_img = output_img.resize(img_size, Image.BICUBIC)

        # Apply palette and save the result
        output_img.putpalette(self.palette)
        output_img = output_img.convert('L')
        result_path = os.path.join(result_dir, os.path.basename(image_path)[:-4] + '.jpg')
        output_img.save(result_path)

    def transform_image(self, img):
        """Transform image using the specified normalization."""
        transforms_list = [transforms.ToTensor(), NormalizeImage(0.5, 0.5)]
        transform_rgb = transforms.Compose(transforms_list)
        image_tensor = transform_rgb(img)
        return torch.unsqueeze(image_tensor, 0)


class NormalizeImage(object):
    """Normalize given tensor into given mean and standard deviation."""

    def __init__(self, mean, std):
        self.normalize_1 = transforms.Normalize([mean], [std])
        self.normalize_3 = transforms.Normalize([mean] * 3, [std] * 3)
        self.normalize_18 = transforms.Normalize([mean] * 18, [std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normalization implemented only for 1, 3, and 18"


# Usage of the class
image_processor = ImageProcessor()

# Process images in the specified directory
image_dir = '/content/inputs/test/cloth'
result_dir = '/content/inputs/test/cloth-mask'

images_list = sorted(os.listdir(image_dir))
for image_name in images_list:
    if image_name.endswith('jpg'):
        image_path = os.path.join(image_dir, image_name)
        image_processor.process_image(image_path, result_dir)
