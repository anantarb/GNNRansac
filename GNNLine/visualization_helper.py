from skimage.draw import line, set_color, circle
import numpy as np
import cv2

def draw_line(data, lX1, lY1, lX2, lY2, clr):

    rr, cc = line(lY1, lX1, lY2, lX2)
    set_color(data, (rr, cc), clr)


def draw_models(labels, clr, data, img_size=64):

    # number of image in batch
    n = labels.shape[0]

    for i in range (n):

        #line
        lY1 = int(labels[i, 0] * img_size)
        lY2 = int(labels[i, 1] * img_size + labels[i, 0] * img_size)
        draw_line(data[i], 0, lY1, img_size, lY2, clr)

    return data


def draw_wpoints(points, data, weights, clrmap, img_size=64):

    color_map = np.arange(256).astype('u1')
    color_map = cv2.applyColorMap(color_map, clrmap)
    color_map = color_map[:,:,::-1]

    n = points.shape[0]
    m = points.shape[2]

    for i in range (0, n):

        s_idx = weights[i].sort(descending=False)[1]
        weights[i] = weights[i] / weights[i].max()

        for j in range(0, m):

            idx = int(s_idx[j])

            # convert weight to color
            clr_idx = float(min(1, weights[i,idx]))
            clr_idx = int(clr_idx * 255)
            clr = color_map[clr_idx, 0] / 255

            # draw point
            r = int(points[i, 0, idx] * img_size)
            c = int(points[i, 1, idx] * img_size)
            rr, cc = circle(r, c, 2)
            set_color(data[i], (rr, cc), clr)

    return data