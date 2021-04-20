import os
import numpy as np
import cv2


def mask_superpose(frame, mask_pred, color=(0, 255, 0)):
    im4show = frame
    mask_pred = np.uint8(mask_pred > 0.3)[:, :, None]
    contours, _ = cv2.findContours(mask_pred.squeeze(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im4show = im4show * (1 - mask_pred) + np.uint8(im4show * mask_pred / 2) + mask_pred * np.uint8(color) * 128

    cv2.drawContours(im4show, contours, -1, color, 2)
    main_contour = contours[np.array([c.shape[0] for c in contours]).argmax()]
    boundrect = cv2.boxPoints(cv2.minAreaRect(main_contour)).astype(np.int64)
    cv2.drawContours(im4show, [boundrect], 0, (0, 0, 255), 3)
    return im4show


def draw_frame_idx(frame, idx):
    H, W, _ = frame.shape
    legend_x, legend_y = W - W//6, H//15
    cv2.putText(frame, '#{}'.format(idx), (legend_x, legend_y+24),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=(0, 255, 255), thickness=3)
    cv2.putText(frame, 'DiMPsuper+AR', (W//15, legend_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 255), thickness=3)
    return frame


def draw_mask(frame, mask, color=(0, 255, 0), idx=0, show=False, save_dir=None):
    out_image = mask_superpose(frame, mask, color)
    out_image = draw_frame_idx(out_image, idx)
    if show:
        cv2.imshow('', out_image)
        cv2.waitKey(0)

    if not save_dir is None:
        cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(idx)), out_image)
