import os
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import math
from os import path as osp


# 投影四点变换——得到变换矩阵
def get_perspective_matrix(image, mode='auto'):
    four_point_dict = {}
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image
    h, w = image.shape[:2]
    if mode == 'auto':
        src = get_contours(image)
        dst = get_dst(src)

    M = cv2.getPerspectiveTransform(src, dst)

    return M


def four_point_perspectivet(image, matrix):
    h, w = image.shape[:2]

    perspective = cv2.warpPerspective(image, matrix, (3 * w, 3 * h), cv2.INTER_LINEAR, borderValue=255)
    perspective = remove_the_blackborder(perspective)
    return perspective

# 去除黑边
def remove_the_blackborder(image):
    # 去除黑边——输入和返回均为图片
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    else:
        image = image
    blur = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    threshold = cv2.threshold(blur, 3, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = threshold[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    # 得到所有黑点的像素坐标
    edges_y, edges_x = np.where(binary_image == 255)
    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom

    left = min(edges_x)
    right = max(edges_x)
    height = top - bottom
    width = right - left
    # 裁剪
    res_image = image[bottom:bottom + height, left:left + width]
    return res_image


def get_dst(src):
    # 求四点中心坐标
    x, y = np.sum(src, axis=0)/4
    # 求
    h1 = math.ceil(np.sqrt((src[1][1] - src[0][1]) ** 2 + (src[1][0] - src[0][0]) ** 2))
    h2 = math.ceil(np.sqrt((src[3][1] - src[2][1]) ** 2 + (src[3][0] - src[2][0]) ** 2))
    w1 = math.ceil(np.sqrt((src[3][1] - src[0][1]) ** 2 + (src[3][0] - src[0][0]) ** 2))
    w2 = math.ceil(np.sqrt((src[1][1] - src[2][1]) ** 2 + (src[1][0] - src[2][0]) ** 2))

    average_w = (w1 + w2) / 2
    average_h = (h1 + h2) / 2
    dst = np.float32([[2*x + 0, 2*y + average_h], [2*x+0, 2*y+0],
                      [2*x + average_w, 2*y+0], [2*x+ average_w, 2*y+average_h]])

    # print(dst)
    return dst


# 调整四角点顺序，按逆时针左下、左上、右上、右下
def order_points(pts):
    #  根据x坐标对进行从小到大的排序
    sort_x = pts[np.argsort(pts[:, 0]), :]
    #  根据点x的坐标排序分别获取所有点中，位于最左侧和最右侧的点
    left = sort_x[:2, :]
    right = sort_x[2:, :]
    # 根据y坐标对左侧的坐标点进行从小到大排序，这样就能够获得左下角坐标点与左上角坐标点
    left = left[np.argsort(left[:, 1])[::-1], :]
    # 根据y坐标对右侧的坐标点进行从小到大排序，这样就能够获得右上角坐标点与右下角坐标点
    right = right[np.argsort(right[:, 1]), :]
    return np.concatenate((left, right), axis=0)


# 得到原始矩形四个变换角点
def get_contours(image):
    # 图像处理——灰化、模糊、腐蚀
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # 边缘检测
    canny = cv2.Canny(gray, 30, 150, 3)
    # 闭运算（链接块）
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # 提取图像轮廓（contours返回点集坐标，hierarchy返回高维信息）
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测

    docCnt = None
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)  # 轮廓多边形拟合
            if cv2.contourArea(cnt) > 400:
                # 轮廓为4个点表示找到纸张 100:
                if len(approx) == 4:
                    docCnt = approx
                    break
    box = []
    for peak in docCnt:
        peak = peak[0]
        cv2.circle(img, tuple(peak), 10, (0, 0, 255))
        box.append([tuple(peak)[0], tuple(peak)[1]])
    src_point = np.float32([box[0], box[1], box[2], box[3]])

    src = order_points(src_point)
    return src


def show_img(img, title="", scale=1):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if np.max(img) < 2:
        img = img * 255
    fig, ax = plt.subplots(ncols=1, figsize=(10*scale, 10*scale))
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')


def count_pixel(image):
    x, y = image.shape
    black = 0
    white = 0
    blue = 0
    for i in range(x):
        for j in range(y):
            if image[i, j] == 0:
                black += 1
            elif image[i, j] > 200:
                white += 1
            else:
                blue += 1
    if black+white > 0:
        rate = black/(black + white)
    else:
        rate = 0
    return rate


def count_waterfilm(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    c = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            res_image = image[int(h*j/4):int(h*(j+1)/4), int(w*(i/4)):int(w*(i+1)/4)]
            c[j][i] = round(count_pixel(res_image), 3)
    return c


if __name__ == '__main__':
    # 设置模型和权重
    config = './work_dirs/fcn_hr18_512x1024_40k_waterfilm.py'
    checkpoint = './checkpoints/latest.pth'
    model = init_segmentor(config, checkpoint, device='cuda:0')
    # 图像根目录
    source = 'work/input'
    images = os.listdir(source)

    for img_name in images:
        # 水膜检测
        img = cv2.imread(os.path.join(source, img_name))
        result = inference_segmentor(model, img)
        model.show_result(img, result, out_file=os.path.join('work/segmentation', img_name), opacity=1)
        # model.show_result(img, result, show=True)
        segmentation = cv2.imread(os.path.join('work/segmentation', img_name))

        # # 倾斜校正
        M = get_perspective_matrix(img)
        perspective = four_point_perspectivet(img, M)
        cv2.imwrite(os.path.join('work/perspective', img_name), perspective)

        count = four_point_perspectivet(segmentation, M)
        cv2.imwrite(os.path.join('work/pixel_count', img_name), count)

        # 计算像素面积
        c = count_waterfilm(os.path.join('work/pixel_count', img_name))
        print(str(img_name) + "的分块区域水膜覆盖率矩阵为：")
        print(c)

        print(str(img_name) + '检测成功')
        # 结果展示
        plt.subplot(221)
        plt.imshow(img[:, :, ::-1])
        plt.title('input')
        plt.subplot(222)
        plt.imshow(perspective[:, :, ::-1])
        plt.title('perspective')
        plt.subplot(223)
        plt.imshow(segmentation[:, :, ::-1])
        plt.subplot(224)
        plt.imshow(count[:, :, ::-1])
        plt.show()





