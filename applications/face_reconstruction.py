import cv2
import torch
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import time
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
from ultralytics import YOLO

# load dl models
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cpu")
midas.to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

segv8 = YOLO('yolov8n-seg.pt')

# capture image
cam = cv2.VideoCapture(0)
time.sleep(1)
_, img = cam.read()
cam.release()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = io.imread('myrt.jpeg')

# compute depth
# get depth
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
output = prediction.cpu().numpy()

# get segmentation
seg = segv8(img)
# find in the boxes the index of the box with class 0 (person)
person_idx = np.where(seg[0].boxes.cls == 0)[0][0]
# get the corresponding mask
person_mask = np.array(seg[0].masks.data[person_idx])

# resize the mask to the size of the depth image
person_mask = cv2.resize(person_mask, (output.shape[1], output.shape[0])).astype(np.uint8)

# remove the points that are too far away (> s.d.)
person_mask[output < output.mean() - 0.2*output.std()] = 0

# now we construct the 3d point cloud with depth and color
# downsize depth image by 10 for fast plotting
person_mask = cv2.resize(person_mask, (int(person_mask.shape[1]/5), int(person_mask.shape[0]/5))).astype(np.uint8)
output = cv2.resize(output, (int(output.shape[1]/5), int(output.shape[0]/5)))
img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)))

# show the 3d point cloud using plotl
xx, yy = np.meshgrid(np.arange(person_mask.shape[1]), np.arange(person_mask.shape[0]))
xx = xx[person_mask == 1]
yy = yy[person_mask == 1]
zz = output[person_mask == 1]

marker_data = go.Scatter3d(
    x=xx.flatten(),
    y=yy.flatten(),
    z=zz.flatten(),
    marker=go.scatter3d.Marker(size=3),
    mode='markers',
    marker_color=img[person_mask == 1].reshape(-1, 3) / 255
)
fig=go.Figure(data=marker_data)
fig.show()