import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import skimage


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])


def convolution(oldimage, kernel):
    # image = Image.fromarray(image, 'RGB')
    image_h = oldimage.shape[0]
    image_w = oldimage.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if len(oldimage.shape) == 3:
        image_pad = np.pad(oldimage, pad_width=( \
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2, \
                                             kernel_w // 2), (0, 0)), mode='constant', \
                           constant_values=0).astype(np.float32)
    elif len(oldimage.shape) == 2:
        image_pad = np.pad(oldimage, pad_width=( \
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2, \
                                             kernel_w // 2)), mode='constant', constant_values=0) \
            .astype(np.float32)

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape)

    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            # sum = 0
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w

    if (h == 0):
        return image_conv[h:, w:w_end]
    if (w == 0):
        return image_conv[h:h_end, w:]

    return image_conv[h:h_end, w:w_end]


def GaussianFilter(image, sigma):
    image = np.asarray(image)
    # print(image)
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
        im_filtered[:, :, c] = convolution(image[:, :, c], gaussian_filter)
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


def canny(img, threshold1, threshold2):
    # Apply Gaussian filter to smooth the image in order to remove the noise
    sigma = 2.5
    img = GaussianFilter(img, sigma)

    img = skimage.color.rgb2gray(img)

    gx = SobelFilter(img, 'x')
    gx = Normalize(gx)
    gy = SobelFilter(img, 'y')
    gy = Normalize(gy)

    # Apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to edge detection


    return img


def print_hi(name):
    img = Image.open('images/messi5.jpg')

    edges = canny(img, 300, 550)
    # edges = GaussianFilter(img, 2.5)
    plt.subplot(121), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    print_hi('PyCharm')
