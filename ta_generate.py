import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def dark_channel(img):
    h, w = img.shape[:2]
    if max(h, w) >= 3000:  
        win_size = 15
    elif max(h, w) >= 1080:  
        win_size = 7
    else:  
        win_size = 3
    b, g, r = cv2.split(img)
    min_img = cv2.min(g, b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win_size, win_size))
    dc_img = cv2.erode(min_img,kernel)
    return dc_img


def get_trans(img, atom, w = 0.95):
    x = img / atom
    t = 1 - w * dark_channel(x)
    return t


def calculate_eta_ratios(t_b, a, lambda_r=700, lambda_g=550, lambda_b=450):
    numerator_r = (-0.00113 * lambda_r + 1.62517) * a[0]#A_b
    denominator_r = (-0.00113 * lambda_b + 1.62517) * a[2]#A_r
    eta_r_over_eta_b = numerator_r / denominator_r
    
    numerator_g = (-0.00113 * lambda_g + 1.62517) * a[0]#A_b
    denominator_g = (-0.00113 * lambda_b + 1.62517) * a[1]#A_g
    eta_g_over_eta_b = numerator_g / denominator_g
    t_r = t_b ** eta_r_over_eta_b
    t_g = t_b ** eta_g_over_eta_b
    
    merged_image = np.stack([t_b, t_g, t_r], axis=-1)
    return merged_image

def calculate_airlight(image, window_size=(25, 25)):

    h, w = image.shape[:2]
    win_h, win_w = window_size

    dark_ = dark_channel(image)

    pixels = []
    for y in range(0, h - win_h + 1, win_h):
        for x in range(0, w - win_w + 1, win_w):

            window = dark_[y:y+win_h, x:x+win_w]

            center_y = y + win_h // 2
            center_x = x + win_w // 2
            pixels.append((window.min(), (center_y, center_x)))
    
    if not pixels:
        return (0.75, 0.75, 0.75)
    
    pixels.sort(reverse=True, key=lambda x: x[0])
    
    top_n = max(int(len(pixels) * 0.01), 1)
    top_coords = [coord for (_, coord) in pixels[:top_n]]
    
    top_rgb_values = []
    for y, x in top_coords:
        if 0 <= y < h and 0 <= x < w:
            top_rgb_values.append(image[y, x])
    
    if not top_rgb_values:
        print("warning")
        return (0.75, 0.75, 0.75)
    
    top_rgb_values = np.array(top_rgb_values)
    airlight = np.mean(top_rgb_values, axis=0)
    
    return tuple(airlight)


def extract_edge_map(img_bgr):
    """
    💡【全新引入】：自适应水下高频轮廓/边缘提取算子（Edge Map, E）
    设计逻辑：
    1. 使用双边滤波器（Bilateral Filter）在保持高频生物边界的同时，滤除低频水体散射和悬浮悬浮颗粒噪声。
    2. 计算自适应 Canny 阈值，精准捕捉不同能见度下的刚性结构特征。
    """
    # 1. 双边滤波保边降噪
    filtered = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=35, sigmaSpace=35)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # 2. 计算自适应大津阈值基准
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # 3. 提取边缘轮廓
    edge = cv2.Canny(gray, lower, upper)

    # 4. 规范化为单通道特征图图 [0, 255]
    return edge


def dehaze_image(image_path, t_save_path, a_save_path, e_save_path, img_save_path=None):
    """
    追加 e_save_path 用来无损存放边缘特征图 E
    """
    im = cv2.imread(image_path)
    if im is None:
        return

    img = im.astype('float64') / 255

    # 1. 物理退化先验计算 (T 和 A)
    atom = calculate_airlight(img)
    trans = get_trans(img, atom)
    trans = np.clip(trans, a_min=0.1, a_max=0.90)
    trans = calculate_eta_ratios(trans, atom)

    # 2. 💡 执行自适应边缘轮廓特征提取 (E)
    edge_map = extract_edge_map(im)

    # 3. 生成物理对齐保存路径
    t_save = os.path.join(t_save_path, os.path.basename(image_path))
    a_save = os.path.join(a_save_path, os.path.basename(image_path))
    e_save = os.path.join(e_save_path, os.path.basename(image_path))

    atom_img = np.full((im.shape[0], im.shape[1], 3), np.multiply(atom, 255), dtype=np.uint8)

    if img_save_path is not None:
        img_save = os.path.join(img_save_path, os.path.basename(image_path))
        cv2.imwrite(img_save, im)

    # 4. 物理数据无损落盘
    cv2.imwrite(t_save, trans * 255)  # 保存透射率 T [B, 3, H, W]
    cv2.imwrite(a_save, atom_img)  # 保存全局大气光 A [B, 3, H, W]
    cv2.imwrite(e_save, edge_map)  # 保存刚性边缘图 E [B, 1, H, W]


def dehaze_V2(originPath, t_save_path, a_save_path, e_save_path):
    image_names = os.listdir(originPath)
    image_paths = [os.path.join(originPath, image_name) for image_name in image_names if
                   image_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(dehaze_image, image_path, t_save_path, a_save_path, e_save_path) for image_path in
                   image_paths]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to batch process UOD physical priors (T, A, E) for datasets.")
    parser.add_argument("input_dir", type=str, help="Main input path for images")
    parser.add_argument("output_dir", type=str, help="Main output path")
    args = parser.parse_args()

    # 自动处理 test 和 train 两个子文件夹 splits
    splits = ["test", "train"]

    for split in splits:
        # Construct the corresponding paths
        # e.g., /opt/data/private/UOD/DUO/images/test
        img_path = os.path.join(args.input_dir, split) 
        
        # e.g., /opt/data/private/UOD/DUO/t/test
        out_t_path = os.path.join(args.output_dir, "t", split)
        
        # e.g., /opt/data/private/UOD/DUO/a/test
        out_a_path = os.path.join(args.output_dir, "a", split)
        out_e_path = os.path.join(args.output_dir, "e", split)  # 💡 自动创建物理边缘保存主轴

        os.makedirs(out_t_path, exist_ok=True)
        os.makedirs(out_a_path, exist_ok=True)
        os.makedirs(out_e_path, exist_ok=True)

        print(f"Processing '{split}' data...")
        print(f" -> Input image path: {img_path}")
        print(f" -> Output T path: {out_t_path}")
        print(f" -> Output A path: {out_a_path}")
        print(f" -> Output E path: {out_e_path}")

        dehaze_V2(img_path, out_t_path, out_a_path, out_e_path)

    print("All data preprocessing complete! T, A, and E maps are perfectly aligned!")