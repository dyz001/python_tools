import numpy as np
import cv2
import os, sys

print(f"numpy version: {np.__version__}")    # numpy 版本
print(f"cv2 version: {cv2.__version__}")   # OpenCV 版本
def extract_ui_elements_transparent(image_path, out_dir="ui_elements_rgba_2", min_area=500):
    """
    从图中提取所有 UI 元素，并把背景设为透明，输出 PNG（含 alpha 通道）
    Args:
      image_path (str): 源图路径
      out_dir (str): 输出目录，会自动创建
      min_area (int): 过滤过小轮廓的面积阈值
    Returns:
      List[Tuple[int,int,int,int]]: 每个元素的 bbox (x,y,w,h)
    """
    # 1. 读图、灰度、边缘、形态学 —— 得到二值掩码
    src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    bboxes = []
    idx = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 3. 提取对应的 ROI 及掩码
        roi_bgr = src[y:y+h, x:x+w]
        mask_full = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask_full, [cnt], -1, 255, -1)
        alpha = mask_full[y:y+h, x:x+w]  # 255/0

        # 4. 合并为 BGRA
        b, g, r = cv2.split(roi_bgr)
        rgba = cv2.merge((b, g, r, alpha))

        # 5. 保存为带透明通道的 PNG
        out_path = os.path.join(out_dir, f"ui_{idx:03d}.png")
        cv2.imwrite(out_path, rgba)
        bboxes.append((x, y, w, h))
        idx += 1

    print(f"already extract {idx} ui elements, folder {out_dir}/")
    return bboxes

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--help" or args[0] == "/?":
        print("extract ui elements from effect image, should easy identify elements in image\n" \
        "-i path of input image\n" \
        "-o directory of output ui elements, default ui_elements\n" \
        "-min min area filter, default 500\n"
        "ex: python extract_elements.py -i source.jpg -o ui -min 400\n" \
        "will read source.jpg and output to ui directory, and dispose the area less than 400\n")
    elif len(args) == 4:
        source = ""
        out_dir = "ui_elements"
        min = 500
        for idx, item in enumerate(args):
            if item == "-i":
                source = args[idx + 1]
            elif item == "-o":
                out_dir = args[idx + 1]
            elif item == "-min":
                min = int(args[idx + 1])
        if not os.path.exists(source):
            print("input error, source file not found")
            exit(10)
        extract_ui_elements_transparent(source, out_dir=out_dir, min_area=min)