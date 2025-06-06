import argparse
import cv2
import numpy as np
import onnxruntime  # 新增
import torch
from tqdm.auto import tqdm
import time
import platform
from pathlib import Path
from det_utils import letterbox, nms, scale_coords
from dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from PIL import Image, ImageDraw, ImageFont
import re
import os
from paddleocr import PaddleOCR
import logging

# 关闭 PaddleOCR 的 DEBUG 日志
logging.getLogger("ppocr").setLevel(logging.INFO)

    
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
label_path = './mylabels.txt'  # 标签


#输入参数为当前文件的绝对路径，输出为当前文件的相对路径
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def load_onnx_model(model_path):
    """加载 ONNX 模型，返回 onnxruntime.InferenceSession 实例"""
    session = onnxruntime.InferenceSession(model_path)
    return session

def preprocess_image(image, cfg, bgr2rgb=True):
    """
    对输入图像进行预处理。
        
    Args:
        image (np.ndarray): 输入图像，形状为[H, W, C] C为3代表RGB三通道。
        cfg (dict): 配置文件 包含输入图像的shape等参数。
        bgr2rgb (bool, optional): 是否将BGR格式转换为RGB格式 默认为True。
        
    Returns:
        tuple: 包含三个元素的元组，分别为：
            - img (np.ndarray): 预处理后的图像，形状为[C, H, W] 数据类型为np.float32，且为连续存储数组。
            - scale_ratio (float): 图像缩放比例。
            - pad_size (tuple): 图像填充大小，形状为(height_pad, width_pad)。
        
    """
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img, scale_ratio, pad_size

def get_labels_from_txt(path):
    labels_dict = {}
    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            labels_dict[i] = line.strip()
    return labels_dict

def draw_bbox(bbox, img0, color, wt, names):
    """在图片上画预测框"""
    det_result_str = ''
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
        img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0]), int(bbox[idx][1] + 64)),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        det_result_str += '{} {} {} {} {} {}\n'.format(
            names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    return img0

def process_result(result_text):
    # 转换为小写以便忽略大小写
    result_text = result_text.lower()
    
    # 处理1：红黄绿灯
    if any(light in result_text for light in ['red_light', 'yellow_light', 'green_light']):
        if 'red_light' in result_text:
            s = "红灯"
            return s
        elif 'yellow_light' in result_text:
            s = "黄灯"
            return s
        elif 'green_light' in result_text:
            s = "绿灯"
            return s
    
    # 处理2：可回收、危险、残余、食物开闭状态
# 处理2：可回收、危险、残余、食物开闭状态
    elif any(state in result_text for state in [
        'recyclable_open', 'recyclable_close',
        'hazardous_open', 'hazardous_close',
        'residual_open', 'residual_close',
        'food_open', 'food_close'
    ]):
        # 判断是否包含 "open"
        if re.search(r'open', result_text):
            s = "未投放垃圾"
        else:
            s = "已投放垃圾"
        
        # 判断垃圾桶类型
        if re.search(r'recyclable', result_text):
            s += "，可回收垃圾桶"
        elif re.search(r'hazardous', result_text):
            s += "，有害垃圾桶"
        elif re.search(r'residual', result_text):
            s += "，其他垃圾桶"
        elif re.search(r'food', result_text):
            s += "，厨余垃圾桶"
        
        return s
    
    # 处理3：floor 和 fire
    elif any(item in result_text for item in ['floor', 'fire']):
        return 0

    # 处理4：ordinary_plp 和 professional_plp
    elif any(plp in result_text for plp in ['ordinary_plp', 'professional_plp']):
        # 解析数量信息
        # print(f"result_text: {result_text}")
        ordinary_match = re.search(r'(\d+)\s*ordinary_plp', result_text)
        professional_match = re.search(r'(\d+)\s*professional_plp', result_text)
        ordinary_count = int(ordinary_match.group(1)) if ordinary_match else 0
        professional_count = int(professional_match.group(1)) if professional_match else 0
        
        s = f"职业人员有{professional_count}人 普通人员有{ordinary_count}人"
        return s
    
    # 默认情况
    else:
        return 1

def detect_firing_floor(pred_all,class_names, labels):
    """检测是否有火灾或楼层"""
    # 分析火灾位置
    if 'floor' in class_names or 'fire' in class_names:
        floors = []  # [(y_min, y_max), ...]
        fires = []   # [(center_y), ...]
        
        # 提取所有楼层和火源的位置信息
        for box in pred_all:
            y1, y2 = box[1], box[3]  # 获取y轴坐标
            cls = int(box[5])
            label = labels[cls]
            
            if label == 'floor':
                floors.append((y1, y2))
            elif label == 'fire':
                center_y = (y1 + y2) / 2
                fires.append(center_y)
        
        # 按y坐标从下到上排序楼层（y值越大表示越下面）
        floors.sort(key=lambda x: x[0], reverse=True)
        
        # 判断每个火源位于哪个楼层
        fire_floors = set()
        for fire_center in fires:
            for floor_idx, (floor_top, floor_bottom) in enumerate(floors, 1):
                if floor_top <= fire_center <= floor_bottom:
                    fire_floors.add(floor_idx)
                    # print(f"火源中心点 {fire_center:.1f} 位于第{floor_idx}层 (楼层范围: {floor_top:.1f}-{floor_bottom:.1f})")
                    break
        
        if fire_floors:
            floor_nums = sorted(list(fire_floors))
            result_text = f"第{', '.join(str(f) for f in floor_nums)}层发生火灾"
            # print(result_text)
            return result_text
        else:
            return None
    
def draw_bbox_with_chinese(bbox, img0, names,ocr,class_names=None, labels=None):
    """在图片上绘制中文标签（无边框和字符背景）"""
    chinese_label="无"
    # 准备中文文本
    chinese_label = process_result(names)
    if not chinese_label:
        s=ocr.ocr(img0)
        texts = [line[1][0] for line in s[0]]  # 只取文本部分
        for text in texts:
            chinese_label = text
             # 如果包含中文，且不包含数字
            if re.search(r'[\u4e00-\u9fff]', chinese_label) and not re.search(r'\d', chinese_label):
                break
        if not any(item in names for item in ['fire']):
           chinese_label +="未发生火灾"
        else:
           result_text=detect_firing_floor(bbox,class_names, labels)
           chinese_label += result_text 
            
    
    
    # for idx, class_id in enumerate(bbox[:, 5]):
    #     if float(bbox[idx][4]) < float(0.05):
    #         continue        
    #     # 使用 PIL 渲染中文文本
    #     img_pil = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(img_pil)
        
    #     # 加载中文字体（需下载支持中文的字体文件，如 simhei.ttf）
    #     font = ImageFont.truetype("simhei.ttf", 140)  # 调整字体大小
    #     text1 = f"{chinese_label}"
    #     # print(f"Drawing text: {text1}at bbox {bbox[idx]}")
    #     # 直接绘制中文文本（无背景）
    #     draw.text((int(bbox[idx][0]), int(bbox[idx][1] + 16)), text1, fill=(0, 0, 255), font=font)
        
    #     # 转换回 OpenCV 格式
    #     img0 = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    #     break
        # # 构建检测结果字符串
        # det_result_str += '{} {} {} {} {} {}\n'.format(
        #     names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    img_pil = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 获取图像尺寸
    img_w, img_h = img_pil.size
    font_size = int(img_h * 0.05)
    font = ImageFont.truetype("simhei.ttf", font_size)  # 调整字体大小
    text1 = f"{chinese_label}"

    
    # 使用 textbbox 获取文字边界框
    bbox = draw.textbbox((0, 0), text1, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # 计算绘制位置（图片底部中央）
    x = (img_w - text_w) // 2
    y = int(img_h * (7 / 8) - text_h / 2)
    
    # 绘制文本
    draw.text((x, y), text1, fill=(0, 0, 255), font=font)   
    img0s = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

    return img0s,chinese_label

def detect_img_onnx(session, raw_img,ocr):
    labels = get_labels_from_txt(label_path)
    cfg = {
        'conf_thres': 0.4,
        'iou_thres': 0.5,
        'input_shape': [640, 640],
    }

    img, scale_ratio, pad_size = preprocess_image(raw_img, cfg)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # 添加 batch 维度

    # ONNX模型输入名
    input_name = session.get_inputs()[0].name
    t1 = time.time()
    pred = session.run(None, {input_name: img})[0]
    t2 = time.time()

    infer_time = t2 - t1
    pred = torch.tensor(pred)

    result_text = "no detections"
    chinese = "无"
    if pred is not None:
        boxout = nms(pred, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
        if boxout and len(boxout[0]) > 0:
            pred_all = boxout[0].numpy()
            scale_coords(cfg['input_shape'], pred_all[:, :4], raw_img.shape, ratio_pad=(scale_ratio, pad_size))
            # draw_bbox(pred_all, raw_img, (0, 255, 0), 2, labels)
            # raw_img = draw_bbox_with_chinese(pred_all, raw_img, labels, labels)  # 使用中文标签绘制
            # print(f"pred_all: {pred_all}, labels: {labels}")
            # 统计每类数量
            classes = pred_all[:, 5].astype(int)
            class_names = [labels[c] for c in classes]
            name_counts = {}
            for name in class_names:
                name_counts[name] = name_counts.get(name, 0) + 1

            result_text = ', '.join([f"{v} {k}" for k, v in name_counts.items()])
            raw_img,chinese = draw_bbox_with_chinese(pred_all, raw_img, result_text,ocr,class_names,labels)
            
            
            
    return chinese, infer_time, raw_img


def img2bytes(image):
    """将图片转换为字节码"""
    return bytes(cv2.imencode('.jpg', image)[1])
def select_source(input_path):
    file_type = 0
    if input_path.lower() == 'camera':
        file_type = 0
    elif os.path.isfile(input_path):
        # 根据文件后缀判断是图片还是视频
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            file_type = 1
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            file_type = 2
        else:
            print("不支持的文件类型")
    else:
        print("文件不存在或路径错误")
    return file_type

def read_and_save_video(save_dir, source=0):
    # 打开摄像头
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 获取视频帧的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 设置视频编码格式和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = str(save_dir / 'output.mp4')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break        
        # 写入帧到输出文件
        out.write(frame)        
        # 显示帧
        cv2.imshow('frame', frame)
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头和文件资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
 
def run2(model='',
    source='',
):
    # 初始化OCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 中文识别

    session = load_onnx_model(model)
    windows = []
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    # 保存目录
    save_dir = increment_path(Path(ROOT / 'runs') / 'exp', exist_ok=False)  # increment run
    (save_dir / 'labels' if False else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    bs = 1  # batch_size
    if webcam:
        dataset = LoadStreams(source, img_size=640, stride=32, auto=True, vid_stride=1)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=640, stride=32, auto=True, vid_stride=1)
    i = 0
    vid_path, vid_writer = [None] * bs, [None] * bs
    for path, im, im0s,s , self in dataset:
        
        # print(f'path: {s}')
        vid_cap = self.cap
        if webcam:  # batch_size >= 1
            p,im0s,frame = path[i],im0s[i].copy(), dataset.count
        else:
            p,  frame  = path, dataset.count
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)
        result_text, infer_time ,im0s= detect_img_onnx(session, im0s,ocr)     
        
        if webcam:
            if platform.system() == "Linux" and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0s.shape[1], im0s.shape[0])
            cv2.imshow(str(p), im0s)
            cv2.waitKey(1)  # 1 millisecond
        print(f'path: {s} Detection time: {infer_time:.3f}s | {result_text}')
        if dataset.mode == 'image':
            cv2.imwrite(save_path, im0s)
                # 打印检测时间和结果
            
            # detect_img(model, im0s,save_path)
        else:  # 'video' or 'stream'
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if 'pbar' in locals() and pbar is not None:
                        pbar.close()  # 关闭之前的进度条
                    pbar = tqdm(total=self.frames, desc=f'video {self.count + 1}/{self.nf}')
                else:  # stream
                    fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0s)
            if vid_cap:
                time.sleep(0.1)
                pbar.update(1)  # 手动更新进度条
        # print(f'p: {p.name}')
        # 
        # # print(f'im0: {im0}')
        # print(f'frame: {frame}')  
    print(f'save_path: {save_dir}')  



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.onnx', help='model path')
    parser.add_argument('--source', type=str, default='data2', help='file/folder/0 for webcam')
    # parser.add_argument('--names', type=str, default='./mylabels.txt', help='class names file')
    opt = parser.parse_args()
    # global label_path
    # label_path = opt.names
    return opt

def main():
    opt = parse_opt()
    run2(**vars(opt))

if __name__ == "__main__":
    main()



# def run3(model='', source='', view_img=False):
#     session = load_onnx_model(model)
#     is_file = Path(source).suffix[1:].lower() in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
#     webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
#     save_dir = increment_path(Path('runs/detect/exp'), exist_ok=False)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     if webcam:
#         dataset = LoadStreams(source, img_size=640, stride=32, auto=True)
#         bs = len(dataset)
#     else:
#         dataset = LoadImages(source, img_size=640, stride=32, auto=True)
#         bs = 1
#     for path, im, im0s, vid_cap, frame_idx in dataset:
#         # im0s 是原图（BGR numpy数组）
#         result_img = detect_img_onnx(session, im0s)
#         if view_img:
#             cv2.imshow(str(path), result_img)
#             cv2.waitKey(1)
#         save_path = save_dir / Path(path).name
#         if dataset.mode == 'image':
#             cv2.imwrite(str(save_path), result_img)
#         else:
#             # 视频处理代码，略，和你之前写的类似
#             pass