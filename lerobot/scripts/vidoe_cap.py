import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 逐帧捕获视频
    ret, frame = cap.read()

    # 如果成功捕获，ret为True
    if not ret:
        print("无法读取帧")
        break

    # 显示当前帧
    cv2.imshow('Video Stream', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和窗口
cap.release()
cv2.destroyAllWindows()