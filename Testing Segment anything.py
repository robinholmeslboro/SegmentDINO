#imports modules
import torch, torchvision, sys, cv2
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

#Shows mask post processing as a new window
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) #generates a random RGB value for the colour
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) #default colour is an azure blue at 60% opacity
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) #no idea what this does and I cant test it yet
    ax.imshow(mask_image)
    #This will be inverted and the background will be blacked out instead (to do, work out how tf to do that)

#shows the bounding box overlayed on the image
def show_box(box, ax):
    x0, y0 = box[0], box[1] #assigns coords to drawing top left corner
    w, h = box[2] - box[0], box[3] - box[1] #finds width & height and assigns
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) #draws the bounding box as a green outline

#displays image on screen as a plt window
def Show_Image():
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off') #disables the graph axis
    plt.show()

image = cv2.imread('/home/rack7/Downloads/tom2set3/scene-%03d00101.jpeg') #loads the image in use
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converts colour order from BGR to RGB (Idk why tbh, I think the input images are RGB)

Show_Image()

sam_checkpoint = "sam_vit_h_4b8939.pth" #sets checkpoint path
model_type = "vit_h" #sets model type to default

#self explanitory
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) #initalises SAM with model
sam.to(device=device) #Assigns sam to device
predictor = SamPredictor(sam)

input_box = np.array([1, 350, 1917, 750])#input box as xyxy
#builds predictor
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

#shows image as plt window with mask and box drawn
def Show_Box_Mask():
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off') #removes axies from window
    plt.show()

Show_Box_Mask()

