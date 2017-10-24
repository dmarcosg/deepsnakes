from PIL import Image, ImageOps
import math
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np

def imrotate(img, angle, fill='black',resample='bilinear'):
    """Rotate the given PIL.Image counter clockwise around its centre by angle degrees.
    Empty region will be padded with color specified in fill."""
    img = Image.fromarray(img)
    theta = math.radians(angle)
    w, h = img.size
    diameter = math.sqrt(w * w + h * h)
    theta_0 = math.atan(float(h) / w)
    w_new = diameter * max(abs(math.cos(theta-theta_0)), abs(math.cos(theta+theta_0)))
    h_new = diameter * max(abs(math.sin(theta-theta_0)), abs(math.sin(theta+theta_0)))
    pad = math.ceil(max(w_new - w, h_new - h) / 2)
    img = ImageOps.expand(img, border=int(pad), fill=fill)
    if resample is 'bicubic':
        img = img.rotate(angle, resample=Image.BICUBIC)
    elif resample is 'bilinear':
        img = img.rotate(angle, resample=Image.BILINEAR)
    elif resample is 'nearest':
        img = img.rotate(angle, resample=Image.NEAREST)
    else:
        print('Dunno what interpolation method ' + resample + ' is.')
    return np.array(img.crop((pad, pad, w + pad, h + pad)))

def plot_snakes(snake,snake_hist,GT,mapE, mapA, mapB, mapK, grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, image, mask):
    # Plot result
    fig0, (ax) = plt.subplots(ncols=1)
    im = ax.imshow(scipy.misc.imresize(image[:,:,:,0],mapE[:, :, 0, 0].shape))
    for i in range(0,len(snake_hist),5):
        ax.plot(snake_hist[i][:, 1], snake_hist[i][:, 0], '--', color=[1-i / len(snake_hist), 1-i / len(snake_hist), i / len(snake_hist)], lw=3)
    ax.plot(GT[:, 1], GT[:, 0], '--', color=[0.2, 1, 0.2], lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '--r', lw=3)
    plt.colorbar(im, ax=ax)
    fig0.suptitle('Image, GT (red) and converged snake (black)', fontsize=20)

    fig0, (ax) = plt.subplots(ncols=1)
    im = ax.imshow(mask[:, :, 0, 0])
    ax.plot(GT[:, 1], GT[:, 0], '--r', lw=3)
    plt.colorbar(im, ax=ax)

    fig1, ax = plt.subplots(ncols=2,nrows=2)
    im0 = ax[0,0].imshow(mapE[:, :, 0, 0])
    plt.colorbar(im0, ax=ax[0,0])
    ax[0, 0].set_title('D')
    im1 = ax[0,1].imshow(mapA[:, :, 0, 0])
    plt.colorbar(im1, ax=ax[0,1])
    ax[0, 1].set_title('alpha')
    im2 = ax[1,0].imshow(mapB[:, :, 0, 0])
    plt.colorbar(im2, ax=ax[1,0])
    ax[1, 0].set_title('beta')
    im3 = ax[1,1].imshow(mapK[:, :, 0, 0])
    plt.colorbar(im3, ax=ax[1,1])
    ax[1, 1].set_title('kappa')
    fig1.suptitle('Output maps', fontsize=20)

    fig2, ax = plt.subplots(ncols=2,nrows=2)
    im0 = ax[0,0].imshow(grads_arrayE[:, :, 0, 0])
    plt.colorbar(im0, ax=ax[0,0])
    ax[0, 0].set_title('D')
    im1 = ax[0,1].imshow(grads_arrayA[:, :, 0, 0])
    plt.colorbar(im1, ax=ax[0,1])
    ax[0, 1].set_title('alpha')
    im2 = ax[1,0].imshow(grads_arrayB[:, :, 0, 0])
    plt.colorbar(im2, ax=ax[1,0])
    ax[1, 0].set_title('beta')
    im3 = ax[1,1].imshow(grads_arrayK[:, :, 0, 0])
    plt.colorbar(im3, ax=ax[1,1])
    ax[1, 1].set_title('kappa')
    fig2.suptitle('Gradient maps', fontsize=20)

    plt.show()