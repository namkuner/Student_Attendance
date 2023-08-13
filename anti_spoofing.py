import torch
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
import torch.nn.functional as F
from src.utility import parse_model_name
import os
import numpy as np
from src.generate_patches import CropImage
MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size
def load_model_anti_spoofing(model_path):
    # define model
    device = torch.device("cuda:0"
                                   if torch.cuda.is_available() else "cpu")
    model_name = os.path.basename(model_path)
    h_input, w_input, model_type, _ = parse_model_name(model_name)
    kernel_size = get_kernel(h_input, w_input,)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)

    # load model weight
    state_dict = torch.load(model_path, map_location=device)
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find('module.') >= 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]
            new_state_dict[name_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    return model

def predict(image, image_bbox,model):
    image_cropper = CropImage()
    h_input, w_input, model_type, scale = parse_model_name("2.7_80x80_MiniFASNetV2.pth")
    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img = image_cropper.crop(**param)
    test_transform = trans.Compose([
        trans.ToTensor(),
    ])
    img = test_transform(img)
    img = img.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        result = model.forward(img)
        result = F.softmax(result).cpu().numpy()
    return result