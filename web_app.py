import cv2
import keras
import numpy as np
import streamlit as st
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageColor
import mediapipe as mp


segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


url = 'https://sphinx-phoenix.github.io/BarberzBuzz/'

st.markdown(f'''
<a href={url}><button style="background-color: #EA4C89;border-radius: 8px;border-style: none;box-sizing: border-box;color:#FFFFFF;">Go Back</button></a>''',unsafe_allow_html=True)






######Load image##########
def load_image(image_file):
    img = Image.open(image_file)
    return img



st.title("Hairstyle generator")

model = keras.models.load_model(
    r'checkpoints/hairnet_matting_30.hdf5')   # Load saved model
model.summary()



#######Functions to change hair color##########
def imShow(image):
    height, width = image.shape[:2]
    resized_image = cv2.resize(
        image, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()


def predict(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)

    pred = model.predict(im)

    mask = pred.reshape((224, 224))

    return mask


def Change_hair_color(image, color):
    global thresh

    #image = cv2.imread(img)
    mask = predict(image)

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    thresh = 0.60  # Threshold used on mask pixels


    blue_mask = mask.copy()
    blue_mask[mask > thresh] = color[0]
    blue_mask[mask <= thresh] = 0

    green_mask = mask.copy()
    green_mask[mask > thresh] = color[1]
    green_mask[mask <= thresh] = 0

    red_mask = mask.copy()
    red_mask[mask > thresh] = color[2]
    red_mask[mask <= thresh] = 0

    blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
    green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
    red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))


    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = blue_mask
    mask_n[:, :, 1] = green_mask
    mask_n[:, :, 2] = red_mask

    alpha = 0.90
    beta = (1.0 - alpha) * 3
    out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)
    return out

    # name = 'test/results/new.jpg'
    # imShow(out)
    # cv2.imwrite(name, out)


# img_path = "soft copy.jpg"
# img = cv2.imread(img_path)

# print(img)

def Change_hair_color_video(image, color):
    mask = predict(image)
    thresh = 0.7  # Threshold used on mask pixels

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    blue_mask = mask.copy()
    blue_mask[mask > thresh] = color[0]
    blue_mask[mask <= thresh] = 0

    green_mask = mask.copy()
    green_mask[mask > thresh] = color[1]
    green_mask[mask <= thresh] = 0

    red_mask = mask.copy()
    red_mask[mask > thresh] = color[2]
    red_mask[mask <= thresh] = 0

    blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
    green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
    red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))

    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = blue_mask
    mask_n[:, :, 1] = green_mask
    mask_n[:, :, 2] = red_mask

    alpha = 0.90
    beta = (1.0 - alpha)*3
    out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return out


color= st.color_picker('Pick A Hair Color', '#00f900')
val=ImageColor.getcolor(color, "RGB")




#color = [124, 100, 250]  # Color to be used on hair

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])


if file is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(file)
    st.write("Original Image")
    st.image(img)
    img1= np.array(img)
    im=Change_hair_color(img1, val)
    st.write("Image with selected hair color")

    st.image(im)




else:
    st.write("Make sure you image is in JPG/PNG/JPEG Format.")


def onChange(pos):
    pass




mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (255, 255, 255)

#cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:
    bg_image = None

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
#
# cv2.namedWindow('img_result')
# cv2.createTrackbar('threshold', 'img_result', 0, 100, onChange)
# cv2.setTrackbarPos('threshold', 'img_result', 50)
#
# cv2.createTrackbar('beta', 'img_result', 0, 50, onChange)
# cv2.setTrackbarPos('beta', 'img_result', 25)
#
# cv2.createTrackbar('R', 'img_result', 0, 255, onChange)
# cv2.createTrackbar('G', 'img_result', 0, 255, onChange)
# cv2.createTrackbar('B', 'img_result', 0, 255, onChange)
#
# while (cap.isOpened()):
#     thresh = cv2.getTrackbarPos('threshold', 'img_result')
#     thresh = thresh / 100
#
#     beta = cv2.getTrackbarPos('beta', 'img_result')
#     beta = beta / 100
#
#     # color[2] = cv2.getTrackbarPos('R', 'img_result')
#     # color[1] = cv2.getTrackbarPos('G', 'img_result')
#     # color[0] = cv2.getTrackbarPos('B', 'img_result')
#
#     ret, frame = cap.read()
#
#     ######################################################
#     # Our initial bg_remover code
#
#     height, width, channel = frame.shape
#     RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = segmentation.process(RGB)
#     mask = results.segmentation_mask
#     Ism = np.stack((mask,) * 3, axis=-1)
#     condition = Ism > 0.6
#     ###############################
#     bg_image = np.zeros(frame.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     #######################################
#     # condition = np. reshape(condition, (height, width,3 ))
#     # background = cv2.resize(background, (width, height))
#     output = np.where(condition, frame, bg_image)
#
#     ###############################################################
#
#     frame = cv2.flip(output, 1)
#
#     rst = Change_hair_color_video(frame, color)
#
#     cv2.imshow('img_result', rst)



# if cv2.waitKey(1) & 0xFF == 27:
#     break
#
# cap.release()
# cv2.destroyAllWindows()

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1) as selfie_segmentation:
        bg_image = None

    height, width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmentation.process(RGB)
    mask = results.segmentation_mask
    Ism = np.stack((mask,) * 3, axis=-1)
    condition = Ism > 0.6
    ###############################
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    #######################################
    # condition = np. reshape(condition, (height, width,3 ))
    # background = cv2.resize(background, (width, height))
    output = np.where(condition, frame, bg_image)
    rst = Change_hair_color_video(output, val)

    ###############################################################

    frame = cv2.flip(rst, 1)



    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')






