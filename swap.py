import cv2
import numpy as np
import face_recognition

def get_landmarks(image):
    face_locations = face_recognition.face_locations(image, model='cnn')
    if not face_locations:
        return None
    face_areas = [(x2 - x1) * (y2 - y1) for (y1, x2, y2, x1) in face_locations]
    largest_idx = np.argmax(face_areas)
    landmarks = face_recognition.face_landmarks(image, face_locations=[face_locations[largest_idx]])
    if not landmarks:
        return None
    lm = landmarks[0]
    keys_order = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
                  'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    points = []
    for key in keys_order:
        points.extend(lm[key])
    return [tuple(p) for p in points]

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst_patch = cv2.warpAffine(src, warp_mat, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst_patch

def warp_triangle(src_img, dst_img, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    t_src_rect = [(p[0] - r1[0], p[1] - r1[1]) for p in t_src]
    t_dst_rect = [(p[0] - r2[0], p[1] - r2[1]) for p in t_dst]
    src_patch = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    warp_patch = apply_affine_transform(src_patch, t_src_rect, t_dst_rect, size)
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), 16, 0)
    dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + warp_patch * mask

def match_color(src_img, dst_img, mask):
    if mask is None or mask.size == 0:
        raise ValueError("Ошибка: маска пуста!")
    if mask.shape[:2] != src_img.shape[:2]:
        mask = cv2.resize(mask, (src_img.shape[1], src_img.shape[0]))
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    src_mean, src_std = cv2.meanStdDev(src_img, mask=mask)
    dst_mean, dst_std = cv2.meanStdDev(dst_img, mask=mask)
    adjusted_img = (dst_img - dst_mean) * (src_std / dst_std) + src_mean
    adjusted_img = np.clip(adjusted_img, 0, 255)
    return adjusted_img.astype(np.uint8)

def face_swap(src_img, dst_img):
    points_src = get_landmarks(src_img)
    points_dst = get_landmarks(dst_img)
    if points_src is None or points_dst is None:
        print("Ошибка: не удалось найти лицевые точки.")
        return None
    points_src = np.array(points_src, np.int32)
    points_dst = np.array(points_dst, np.int32)
    hull_index = cv2.convexHull(points_dst, returnPoints=False)
    hull_src = [points_src[idx[0]] for idx in hull_index]
    hull_dst = [points_dst[idx[0]] for idx in hull_index]
    mask = np.zeros(dst_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(hull_dst), 255)
    r = cv2.boundingRect(np.float32([hull_dst]))
    warped_face = np.copy(dst_img)
    subdiv = cv2.Subdiv2D(r)
    for p in hull_dst:
        subdiv.insert((int(p[0]), int(p[1])))
    triangles = np.array(subdiv.getTriangleList(), dtype=np.float32)
    triangle_indices = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = []
        for p in pts:
            for i, hp in enumerate(hull_dst):
                if abs(p[0] - hp[0]) < 1.0 and abs(p[1] - hp[1]) < 1.0:
                    indices.append(i)
                    break
        if len(indices) == 3:
            triangle_indices.append(indices)
    for indices in triangle_indices:
        t_src = [hull_src[i] for i in indices]
        t_dst = [hull_dst[i] for i in indices]
        warp_triangle(src_img, warped_face, t_src, t_dst)
    mask = cv2.GaussianBlur(mask, (15, 15), 10)
    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)
    output = cv2.seamlessClone(np.uint8(warped_face), dst_img, mask, center, cv2.NORMAL_CLONE)
    return output

if __name__ == '__main__':
    src_img = cv2.imread('img37.png')
    dst_img = cv2.imread('img36.png')
    if src_img is None or dst_img is None:
        print("Ошибка загрузки изображений.")
        exit()
    result = face_swap(src_img, dst_img)
    if result is not None:
        cv2.imwrite('face_swapped.jpg', result)
        print("Готово! Изображение сохранено как face_swapped.jpg")
    else:
        print("Не удалось выполнить замену лица.")
