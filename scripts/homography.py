import cv2
import numpy as np

def get_transformed_corners(dimensions: tuple[int, int], H):
    top_left = H @ np.array([[0], [0], [1]])
    top_right = H @ np.array([[dimensions[1]], [0], [1]])
    bottom_left = H @ np.array([[0], [dimensions[0]], [1]])
    bottom_right = H @ np.array([[dimensions[1]], [dimensions[0]], [1]])
    top_left /= top_left[2][0]
    top_right /= top_right[2][0]
    bottom_left /= bottom_left[2][0]
    bottom_right /= bottom_right[2][0]
    return top_left, top_right, bottom_left, bottom_right

def compute_translation(transformed_corners):
    return np.array([[1, 0, -min([corner[0][0] for corner in transformed_corners])],
                     [0, 1, -min([corner[1][0] for corner in transformed_corners])],
                     [0, 0, 1]])

def homography_one_image(im, H):
    new_corners_im1 = get_transformed_corners(im.shape, H)

    #t = compute_translation(new_corners_im1)

    #bottom_padding = max(0, max([int(c[1][0]) for c in new_corners_im1]) - im.shape[0])
    #new_shape = (im.shape[1] + int(t[0][2]), im.shape[0] + int(t[1][2]) + bottom_padding)

    return cv2.warpPerspective(im, H, (int(im.shape[1]), int(im.shape[0])),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))

def homography_two_images(im_1, im_2, H):
    new_corners_im1 = get_transformed_corners(im_1.shape, H)

    t = compute_translation(new_corners_im1)

    bottom_padding = max(0, max([int(c[1][0]) for c in new_corners_im1]) - im_2.shape[0])
    new_shape = (im_2.shape[1] + int(t[0][2]), im_2.shape[0] + int(t[1][2]) + bottom_padding)

    im_1 = cv2.warpPerspective(im_1, t @ H, new_shape,
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))

    im_2 = cv2.warpPerspective(im_2, t, new_shape,
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))

    # masks: 1 where pixel valid, 0 where black
    m1 = (np.sum(im_1, axis=2) > 0).astype(np.float32)
    m2 = (np.sum(im_2, axis=2) > 0).astype(np.float32)

    # weighted sum
    sum_img = im_1.astype(np.float32) * m1[..., None] + im_2.astype(np.float32) * m2[..., None]
    weights = (m1 + m2)[..., None]

    # avoid division by 0
    out = np.zeros_like(sum_img, dtype=np.float32)
    valid = (weights[..., 0] > 0)
    out[valid] = sum_img[valid] / weights[valid]

    return np.clip(out, 0, 255).astype(np.uint8)

def get_epipolar_line(p1, F, max_x) -> tuple[tuple[int, int], tuple[int, int]]:
    # p2t F p1 = 0 -> p2t X = 0
    X = F @ p1
    p_low = (0, round(float(-X[2]/X[1])))
    p_high = (max_x, round(float((-X[0]*max_x-X[2])/X[1])))
    return p_low, p_high

def draw_point(img, pos):
    return cv2.circle(img, pos, 8, (0, 0, 255), -1)

def draw_line(img, p1, p2):
    return cv2.line(img, p1, p2, (0, 0, 255), 5)

im_1 = cv2.imread("assets/rectification/img_right.jpg")
# im_2 = cv2.imread("assets/homography/img_34.png")

H = np.array([[1.156, 0.157, -150.34],
              [0.0374, 1.0078, -26.91],
              [0.00035, 0.0000349, 0.81749]])

F = np.array([[-1.41539e-08, 2.82758e-07, -1.6399e-05],
              [-8.97881e-08, 7.82147e-09, -0.0009103],
              [-3.28468e-05, 0.000803705, 0.0285408]]).T

p1 = (459, 177, 1)
p1, p2 = get_epipolar_line(p1, F, im_1.shape[1])

# out = homography_two_images(im_1, im_2, H)
# out = homography_one_image(im_1, H)
#out = draw_point(im_1, (459, 177))
out = draw_line(im_1, p1, p2)

cv2.imwrite("out.png", out)
print("Written file out.png")