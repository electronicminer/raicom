import cv2
import numpy
import numpy as np
from numba import njit
 
 
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
    cv2.namedWindow('HSV_Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV_Trackbars', 600, 400)
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

# 修改XunX函数中的相关部分：
def XunX(img, SX):
    frameBGR = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    colorLow, colorHigh = get_hsv_values()
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    height = mask.shape[0]
    half_height = height // 2
    mask[half_height:, :] = 0
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    mask = cv2.bitwise_not(mask)

    scan_rows = mask.shape[0]
    left = np.zeros(scan_rows, dtype=np.int32)
    right = np.zeros(scan_rows, dtype=np.int32)
    leftb = np.zeros(scan_rows, dtype=np.int32)
    rightb = np.zeros(scan_rows, dtype=np.int32)
    SXp = np.zeros(scan_rows, dtype=np.int32)

    valid_len, SX_now = scan_lines(mask, SX, left, right, leftb, rightb, SXp)

    # 截取有效部分
    left = left[:valid_len]
    right = right[:valid_len]
    leftb = leftb[:valid_len]
    rightb = rightb[:valid_len]
    SXp = SXp[:valid_len]

    # 后续逻辑保持不变
    # ...existing code...

    medim = (leftb + rightb) / 2
    img_display = img.copy()
    half_height = img.shape[0] // 2

    for k in range(valid_len-1, -1, -1):
        current_y = 479 - k
        if current_y < half_height:
            point = (int(medim[k]), current_y)
            point3 = (int(leftb[k]), current_y)
            point1 = (int(rightb[k]), current_y)
            cv2.circle(img_display, point, 2, (0, 255, 0), -1)
            cv2.circle(img_display, point1, 2, (0, 0, 255), -1)
            cv2.circle(img_display, point3, 2, (255, 0, 0), -1)

    center_line_avg = np.mean(medim)
    direction = ""
    if center_line_avg < 300:
        direction = "左转"
    elif center_line_avg > 340:
        direction = "右转"
    else:
        direction = "直行"
    print(f" 方向: {direction}")
    cv2.imshow('lane_detection', img_display)

    return SX_now, (leftb + rightb) / 2, SXp

@njit
def scan_lines(mask, SX, left, right, leftb, rightb, SXp):
    zz = 0
    yy = mask.shape[1] - 1
    SX_now = SX
    idx = 0
    for i in range(mask.shape[0], 1, -1):
        left_found = False
        right_found = False

        # 扫左线
        for j in range(SX_now + 1, 0, -1):
            if j >= 1:
                if (mask[i - 1, j] == 0 and
                    (j == 1 or mask[i - 1, j - 1] == 255 or
                     (j >= 2 and mask[i - 1, j - 2] == 255))):
                    left[idx] = j
                    leftb[idx] = j
                    left_found = True
                    zz = j
                    break
        if not left_found:
            left[idx] = 0
            leftb[idx] = 0

        # 扫右线
        for j1 in range(SX_now + 2, mask.shape[1]-1, 1):
            if j1 <= mask.shape[1]-2:
                if (mask[i - 1, j1] == 0 and
                    (j1 == mask.shape[1]-2 or mask[i - 1, j1 + 1] == 255 or
                     (j1 <= mask.shape[1]-3 and mask[i - 1, j1 + 2] == 255))):
                    right[idx] = j1
                    rightb[idx] = j1
                    right_found = True
                    yy = j1
                    break
        if not right_found:
            right[idx] = mask.shape[1]-1
            rightb[idx] = mask.shape[1]-1

        # 提前结束扫描的条件
        if (left[idx] == 0 and right[idx] != mask.shape[1]-1 and
            i > 2 and mask[i-2, :10].sum() == 0):
            break

        # 更新扫描中心点
        if left_found and right_found:
            SX_now = int((left[idx] + right[idx]) / 2)
        elif left_found:
            SX_now = int(left[idx] + 100)
        elif right_found:
            SX_now = int(right[idx] - 100)
        else:
            SX_now = mask.shape[1] // 2

        SXp[idx] = SX_now
        idx += 1
    return idx, SX_now  # 返回有效长度和SX_now
 
mm = 320#扫线开始的坐标

#使用摄像头
# cap = cv2.VideoCapture(0)
#使用本地视频
video_path = r"巡线.mp4"  # 确保视频文件在正确的路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: 无法打开视频文件")
    exit()

mm = 320  # 扫线开始的坐标

# 创建HSV滑动条
create_hsv_trackbars()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("视频读取完成")
        break
        
    try:
        frame = cv2.resize(frame, (640, 480))
        mm, XB, SXp = XunX(frame, mm)
        
        # 添加适当的延时控制帧率
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    except Exception as e:
        print(f"处理帧时发生错误: {e}")
        continue

cap.release()
cv2.destroyAllWindows()

