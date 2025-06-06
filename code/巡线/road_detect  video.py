import cv2
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def nothing(*arg):
    pass
 
 
icol = (18, 0, 196, 36, 255, 255)
 
#path = "test/cruise/"
# Show the original image.
#frame = path+str(961)+'.jpg'
#frame = cv2.imread(frame)
l = [16, 45, 65]  # [17, 55, 128]#阈值
h = [44, 255, 255]  # [24, 255, 255]#阈值
 
def zh_ch(string):
    return string.encode("gbk").decode('UTF-8', errors='ignore')
 
def create_hsv_trackbars():
    cv2.namedWindow('HSV_Trackbars')
    # H范围是0-180，S和V范围是0-255
    cv2.createTrackbar('H_min', 'HSV_Trackbars', 0, 180, nothing)
    cv2.createTrackbar('H_max', 'HSV_Trackbars', 180, 180, nothing)
    cv2.createTrackbar('S_min', 'HSV_Trackbars', 0, 255, nothing)
    cv2.createTrackbar('S_max', 'HSV_Trackbars', 255, 255, nothing)
    cv2.createTrackbar('V_min', 'HSV_Trackbars', 0, 255, nothing)
    cv2.createTrackbar('V_max', 'HSV_Trackbars', 142, 255, nothing)

def get_hsv_values():
    h_min = cv2.getTrackbarPos('H_min', 'HSV_Trackbars')
    h_max = cv2.getTrackbarPos('H_max', 'HSV_Trackbars')
    s_min = cv2.getTrackbarPos('S_min', 'HSV_Trackbars')
    s_max = cv2.getTrackbarPos('S_max', 'HSV_Trackbars')
    v_min = cv2.getTrackbarPos('V_min', 'HSV_Trackbars')
    v_max = cv2.getTrackbarPos('V_max', 'HSV_Trackbars')
    return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

# 在文件开头导入PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 添加一个绘制中文的函数
def draw_chinese_text(img, text, position, color):
    # 创建PIL图像
    img_pil = Image.fromarray(img)
    
    # 创建draw对象
    draw = ImageDraw.Draw(img_pil)
    
    # 指定字体和大小（需要系统安装了相应的字体）
    fontpath = "C:/Windows/Fonts/simhei.ttf"  # 使用系统黑体
    font = ImageFont.truetype(fontpath, 32)
    
    # 绘制文字
    draw.text(position, text, font=font, fill=color[::-1])  # PIL顺序是RGB，需要转换BGR顺序
    
    # 转换回OpenCV格式
    return np.array(img_pil)

# 修改XunX函数中的相关部分：
def XunX(img,SX):
    # cv2.imshow('frame', img)
    
    # 高斯模糊
    frameBGR = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    
    # 使用滑动条获取HSV阈值
    colorLow, colorHigh = get_hsv_values()
    
    # 二值化
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    
    # 获取图像高度
    height = mask.shape[0]
    half_height = height // 2
    
    # 将下半部分填充为白色（255）
    mask[half_height:, :] = 0

    # 显示原始二值化结果
    # cv2.imshow('mask-original', mask)
    
    # 形态学处理 - 减小核的大小以保留更多细节
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    
    # 反转mask使黑线变为白色 (可选，取决于你的算法期望的输入)
    mask = cv2.bitwise_not(mask)
    
    # 显示处理后的二值化结果
    # cv2.imshow('mask-processed', mask)
    # Find_Line(mask)
 
    left = np.array([])
    right = np.array([])
    leftb = np.array([])
    rightb = np.array([])
    medim = np.array([])
    # len = []#记录边线丢失
    le = [[]]
    ri = [[]]
    zz = 0  # 记录左缺陷时上一个坐标
    yy = 639
    SX = SX  # 开始扫线时的y坐标
    SXp = np.array([])
    SS = 0
    left_up = left_down = right_up = right_down = 0
    for i in range(480, 1, -1):
        left_found = False
        right_found = False
        
        # 扫左线 - 改进扫描方式
        for j in range(SX + 1, 0, -1):
            if j >= 1:  # 防止越界
                # 检测跳变点，增加容错
                if (mask[i - 1][j] == 0 and 
                    (j == 1 or mask[i - 1][j - 1] == 255 or 
                     (j >= 2 and mask[i - 1][j - 2] == 255))):  # 增加容错检测
                    # 记录左跳变点的列值
                    left = np.append(left, j)
                    leftb = np.append(leftb, j)
                    left_found = True
                    zz = j
                    break
        
        # 如果没找到左边界
        if not left_found:
            lk = 0
            left = np.append(left, lk)
            leftb = np.append(leftb, lk)
            
        # 扫右线 - 改进扫描方式
        for j1 in range(SX + 2, 639, 1):
            if j1 <= 638:  # 防止越界
                # 检测跳变点，增加容错
                if (mask[i - 1][j1] == 0 and 
                    (j1 == 638 or mask[i - 1][j1 + 1] == 255 or 
                     (j1 <= 637 and mask[i - 1][j1 + 2] == 255))):  # 增加容错检测
                    # 记录右跳变点的列值
                    right = np.append(right, j1)
                    rightb = np.append(rightb, j1)
                    right_found = True
                    yy = j1
                    break
                    
        # 如果没找到右边界
        if not right_found:
            lk = 639
            right = np.append(right, lk)
            rightb = np.append(rightb, lk)

        # 提前结束扫描的条件修改
        if (left[480 - i] == 0 and right[480 - i] != 639 and 
            i > 2 and np.sum(mask[i-2, :10]) == 0):  # 检查左侧边缘是否确实没有线
            break
            
        # 更新扫描中心点
        if left_found and right_found:
            SX = int((left[480 - i] + right[480 - i]) / 2)
        elif left_found:
            SX = int(left[480 - i] + 100)  # 如果只找到左边界，向右偏移
        elif right_found:
            SX = int(right[480 - i] - 100)  # 如果只找到右边界，向左偏移
        else:
            SX = 320  # 如果都没找到，回到中心
            
        SXp = np.append(SXp, SX)

    # 找拐点
    sm = 30
    for i in range(len(left) - 10):
        if (left[i] == 0 and left[i + 3] == 0 and left[i + 5] > 0 and left[i + 10] > 0):
            left_up = 1
            left_up1 = (i + min(len(left[i+2:]),sm), left[i + min(len(left[i+2:]),sm)])  # 480 - i-sm
        if (left[i] > 0 and left[i + 3] > 0 and left[i + 5] == 0 and left[i + 10] == 0):
            left_down = 1
            left_down1 = (i - min(i,sm), left[i - min(i,sm)])  # 480 - i+sm
        if (right[i] == 639 and right[i + 3] == 639 and right[i + 5] < 639 and right[i + 10] <= 639):
            right_up = 1
            right_up1 = (i + min(len(left[i+2:]),sm), right[i + min(len(left[i+2:]),sm)])
        if (right[i] < 639 and right[i + 3] < 639 and right[i + 5] == 639 and right[i + 10] == 639):
            right_down = 1
            right_down1 = (i - min(i,sm), right[i - min(i,sm)])  # -1
    # 判断元素：补线操作
    # print(left_up, left_down, right_up, right_down)
    if (left_up and not left_down):
        left_down1 = (0, left_up1[1])
    elif (not left_up and left_down):
        left_up1 = (479, left_down1[1])
    if (right_up and not right_down):
        right_down1 = (0, right_up1[1])
    elif (not right_up and right_down):
        right_up1 = (479, right_down1[1])
    # 左右的上下拐点同时出现
    #if (left_up and right_up) or (left_down and right_down):
    left_up = left_down = right_up = right_down = 0
    left_up1 = left_down1 = right_up1 = right_down1 = (0,0)  # 添加这行初始化
    
    # 修改拐点判断部分
    if (left_up or right_up) or (left_down or right_down):
        try:
            for k in range(len(leftb)):
                if (k >= left_down1[0] and k <= left_up1[0]) or (k <= left_down1[0] and k >= left_up1[0]):
                    leftb[k] = ((left_up1[1] - left_down1[1]) / (left_up1[0] -
                                                                left_down1[0])) * (k - left_up1[0]) + left_up1[1]
                if (k >= right_down1[0] and k <= right_up1[0]) or (k <= right_down1[0] and k >= right_up1[0]):
                    rightb[k] = ((right_up1[1] - right_down1[1]) / (right_up1[0] -
                                                                    right_down1[0])) * (k - right_up1[0]) + right_up1[1]
        except Exception as e:
            print(f"处理拐点时发生错误: {e}")
            pass  # 错误处理，防止程序崩溃
    # # 同时出现左上下，或右上下：
    # if (left_up and left_down) or (right_up and right_down):
    #     # if(right_up1[0] - left_up1[0] <= 20):
    #     for k in range(len(leftb)):
    #         if (k >= left_down1[0] and k <= left_up1[0]) or (k <= left_down1[0] and k >= left_up1[0]):
    #             leftb[k] = ((left_up1[1] - left_down1[1]) / (left_up1[0] -
    #                                                          left_down1[0])) * (k - left_up1[0]) + left_up1[1]
    #         if (k >= right_down1[0] and k <= right_up1[0]) or (k <= right_down1[0] and k >= right_up1[0]):
    #             rightb[k] = ((right_up1[1] - right_down1[1]) / (right_up1[0] -
    #                                                             right_down1[0])) * (k - right_up1[0]) + right_up1[1]
    # left_up = left_down = right_up = right_down = 0
    #  if
 
    # 修改找拐点和补线部分
    sm = 30
    last_valid_left = None  # 记录最后一个有效的左边界点
    continuous_zero_count = 0  # 连续检测不到左边界的计数
    
    # 先遍历一遍找到有效的左边界点
    for i in range(len(left)):
        if left[i] > 0:
            last_valid_left = (i, left[i])
            continuous_zero_count = 0
        else:
            continuous_zero_count += 1
            
        # 当连续5个点都没有检测到左边界,且右边界存在,说明进入转弯
        if continuous_zero_count >= 5 and last_valid_left is not None and right[i] != 639:
            # 使用圆弧插值进行补线
            start_idx = last_valid_left[0]
            for k in range(start_idx, min(start_idx + 30, len(left))):
                if k < len(left):
                    # 根据右边界的弯曲程度动态调整补线
                    curve_offset = (right[k] - right[start_idx]) * 0.5  # 根据右边界变化计算偏移量
                    predicted_left = max(0, last_valid_left[1] - curve_offset)
                    leftb[k] = predicted_left
                    left[k] = predicted_left
        
        # 如果重新检测到左边界,更新last_valid_left
        elif left[i] > 0:
            last_valid_left = (i, left[i])
            continuous_zero_count = 0

    # 平滑处理补线结果
    if len(leftb) > 5:
        for i in range(2, len(leftb)-2):
            if leftb[i] == 0:  # 只平滑补线部分
                leftb[i] = (leftb[i-2] + leftb[i-1] + leftb[i+1] + leftb[i+2]) / 4

    medim = (leftb + rightb) / 2
    #  img = [mask,mask3,mask4,mask5]
    pl = ['left', 'right', 'l&r', 'no']
    o = 0
    img_display = img.copy()  # 创建原始图像的副本用于显示
    
    cg = len(left)-1
    # print(cg,len(leftb),len(right))

    half_height = img.shape[0] // 2  # 获取图像高度的一半
    
    for k in range(cg, -1, -1):
        current_y = 479 - k  # 当前要绘制的y坐标
        
        # 只绘制上半部分(y坐标小于half_height的部分)
        if current_y < half_height:
            point = (int(medim[k]), current_y)    # 中线点
            point3 = (int(leftb[k]), current_y)   # 左边线点
            point1 = (int(rightb[k]), current_y)  # 右边线点
            
            # 使用不同颜色绘制三条线
            cv2.circle(img_display, point, 2, (0, 255, 0), -1)    # 中线用绿色
            cv2.circle(img_display, point1, 2, (0, 0, 255), -1)   # 右线用红色
            cv2.circle(img_display, point3, 2, (255, 0, 0), -1)   # 左线用蓝色
    
    
    # 同时显示二值化图像用于调试
    # cv2.imshow('binary', mask)
     # 计算中线的平均位置
    center_line_avg = np.mean(medim)
    
    # 判断方向
    direction = ""
    if center_line_avg < 300:  # 中线偏左
        direction = "左转"
    elif center_line_avg > 340:  # 中线偏右
        direction = "右转"
    else:
        direction = "直行"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (30, 30)  # 文本位置在左上角
    font_scale = 1
    font_thickness = 2
    
    # 根据方向使用不同颜色
    if direction == "右转":
        color = (255, 0, 0)  # 蓝色
    elif direction == "左转":
        color = (0, 0, 255)  # 红色
    else:  # 直行
        color = (0, 255, 0)  # 绿色
        
    # 使用新的绘制中文函数
    img_display = draw_chinese_text(img_display, direction, (30, 30), color)

    # 显示带有彩色线的原始图像
    cv2.imshow('lane_detection', img_display)

    return SX, (leftb + rightb) / 2, SXp, img_display  # 添加 img_display 到返回值
 
mm = 320#扫线开始的坐标
#使用摄像头
cap = cv2.VideoCapture(0)

#使用本地视频
video_path = "巡线.mp4"  # 确保视频文件在正确的路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: 无法打开视频文件")
    exit()

mm = 320  # 扫线开始的坐标

# 创建HSV滑动条
create_hsv_trackbars()

# 设置保存视频的参数
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编码格式
out_fps = 30  # 输出视频的帧率
out_size = (640, 480)  # 输出视频的尺寸
out = cv2.VideoWriter('lane_detection_output.avi', fourcc, out_fps, out_size)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("视频读取完成")
        break
        
    try:
        frame = cv2.resize(frame, (640, 480))
        mm, XB, SXp, processed_frame = XunX(frame, mm)  # 接收返回的图像
        
        # 保存处理后的帧
        out.write(processed_frame)
        
        # 添加适当的延时控制帧率
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"处理帧时发生错误: {e}")
        continue

cap.release()
cv2.destroyAllWindows()

