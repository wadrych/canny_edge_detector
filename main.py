import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import skimage


def Convolution(oldimage, kernel):
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if len(oldimage.shape) == 3:
        image_pad = np.pad(oldimage, pad_width=(
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2,
                                             kernel_w // 2), (0, 0)), mode='constant',
                           constant_values=0).astype(np.float32)
    elif len(oldimage.shape) == 2:
        image_pad = np.pad(oldimage, pad_width=(
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2,
                                             kernel_w // 2)), mode='constant',
                           constant_values=0).astype(np.float32)

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape)

    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w

    if h == 0:
        return image_conv[h:, w:w_end]
    if w == 0:
        return image_conv[h:h_end, w:]

    return image_conv[h:h_end, w:w_end]


def GaussianFilter(image, sigma):
    image = np.asarray(image)
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2

    im_filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        im_filtered[:, :, c] = Convolution(image[:, :, c], gaussian_filter)
    return im_filtered.astype(np.uint8)


def SobelFilter(img, direction):
    if direction == 'x':
        Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        Res = ndimage.convolve(img, Gx)
    if direction == 'y':
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
        Res = ndimage.convolve(img, Gy)

    return Res


def Normalize(img):
    # img = np.multiply(img, 255 / np.max(img))
    img = img / np.max(img)
    return img


def NonMaxSupWithoutInterpol(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if (-22.5 <= Grad[i, j] <= 22.5) or (-157.5 >= Grad[i, j] >= 157.5):
                if (Gmag[i, j] > Gmag[i, j + 1]) and (Gmag[i, j] > Gmag[i, j - 1]):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if (22.5 <= Grad[i, j] <= 67.5) or (-112.5 >= Grad[i, j] >= -157.5):
                if (Gmag[i, j] > Gmag[i + 1, j + 1]) and (Gmag[i, j] > Gmag[i - 1, j - 1]):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if (67.5 <= Grad[i, j] <= 112.5) or (-67.5 >= Grad[i, j] >= -112.5):
                if (Gmag[i, j] > Gmag[i + 1, j]) and (Gmag[i, j] > Gmag[i - 1, j]):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0
            if (112.5 <= Grad[i, j] <= 157.5) or (-22.5 >= Grad[i, j] >= -67.5):
                if (Gmag[i, j] > Gmag[i + 1, j - 1]) and (Gmag[i, j] > Gmag[i - 1, j + 1]):
                    NMS[i, j] = Gmag[i, j]
                else:
                    NMS[i, j] = 0

    return NMS


def TrackByHysteresis(img, highThresholdRatio, lowThresholdRatio):
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if GSup[i, j] > highThreshold:
                GSup[i, j] = 1
            elif GSup[i, j] < lowThreshold:
                GSup[i, j] = 0
            else:
                if ((GSup[i - 1, j - 1] > highThreshold) or
                        (GSup[i - 1, j] > highThreshold) or
                        (GSup[i - 1, j + 1] > highThreshold) or
                        (GSup[i, j - 1] > highThreshold) or
                        (GSup[i, j + 1] > highThreshold) or
                        (GSup[i + 1, j - 1] > highThreshold) or
                        (GSup[i + 1, j] > highThreshold) or
                        (GSup[i + 1, j + 1] > highThreshold)):
                    GSup[i, j] = 1

    GSup = (GSup == 1) * GSup

    return GSup


def Canny(img, high_threshold, low_threshold):
    # Apply Gaussian filter to smooth the image in order to remove the noise
    sigma = 1.5
    img = GaussianFilter(img, sigma)
    img = skimage.color.rgb2gray(img)

    # Finding the intensity gradient of the image
    gx = SobelFilter(img, 'x')
    gx = Normalize(gx)
    gy = SobelFilter(img, 'y')
    gy = Normalize(gy)

    gradient = np.degrees(np.arctan2(gy, gx))

    # Apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to
    # edge detection
    mag = np.hypot(gx, gy)

    img = NonMaxSupWithoutInterpol(mag, gradient)
    img = Normalize(img)

    # Apply double threshold to determine potential edges Track edge by hysteresis: Finalize the detection of edges
    # by suppressing all the other edges that are weak and not connected to strong edges.
    img = TrackByHysteresis(img, high_threshold, low_threshold)

    return img


def Main():
    img = Image.open('images/xanax5.jpeg')
    edges = Canny(img, 0.09, 0.05)

    plt.subplot(121), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap=plt.get_cmap('gray'))
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    Main()
