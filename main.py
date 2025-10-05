from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from media.sensor import *
from media.display import *
import time
import utime
import nncase_runtime as nn
import ulab.numpy as np
import gc
import sys
import aicube
from machine import PWM, FPIOA, Pin
from pid import PID
from kalman_filter import KalmanFilter

# 导入GPIO控制模块
from gpio_control import init_gpio_pins, control_gpios

# 全局变量
person_det = None
pl = None
sensor = None
gpio_objs = None  # GPIO控制对象

# 发射控制相关变量
in_center_start_time = None  # 开始位于中心区域的时间
CENTER_DELAY = 3000  # 中心区域停留时间阈值（毫秒）
last_launch_time = 0  # 上次发射时间，避免频繁触发
LAUNCH_INTERVAL = 5000  # 发射间隔（毫秒）

# 通信协议相关
from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart
uart = YbUart(baudrate=115200)
pto = YbProtocol()

###############################舵机配置#####################################################
min_duty = 1638      # 2.5% 占空比
max_duty = 8192      # 12.5% 占空比
mid_duty = 4915      # 7.5% 占空比 (中间位置)
pwm_lr = None        # 左右舵机
pwm_ud = None        # 上下舵机

# 记录舵机上次位置，用于限制转动速度
last_lr_duty = mid_duty
last_ud_duty = mid_duty

###############################检测配置#####################################################
DETECT_WIDTH = 640
DETECT_HEIGHT = 480

# 中心区域设置
CENTER_REGION_RATIO = 0.2  
center_width = int(DETECT_WIDTH * CENTER_REGION_RATIO)
center_height = int(DETECT_HEIGHT * CENTER_REGION_RATIO)
center_x = DETECT_WIDTH // 2
center_y = DETECT_HEIGHT // 2

###############################PID配置######################################
lr_kp = 0.12    # 左右比例系数
lr_ki = 0.01    # 左右积分系数
lr_kd = 0.001   # 左右微分系数
lr_max_out = 100  # 左右最大输出

ud_kp = 0.12    # 上下比例系数
ud_ki = 0.01    # 上下积分系数
ud_kd = 0.001   # 上下微分系数
ud_max_out = 100  # 上下最大输出

pid_lr = PID(lr_kp, lr_ki, lr_kd, lr_max_out, lr_max_out)
pid_ud = PID(ud_kp, ud_ki, ud_kd, ud_max_out, ud_max_out)

###############################卡尔曼滤波器######################################
kalman_lr = KalmanFilter(process_noise=0.1, measurement_noise=0.6, error_estimate=1.0)
kalman_ud = KalmanFilter(process_noise=0.1, measurement_noise=0.6, error_estimate=1.0)

# 自定义人体检测类
class PersonDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, labels, anchors, confidence_threshold=0.2, nms_threshold=0.5, nms_option=False, strides=[8, 16, 32], rgb888p_size=[224, 224], display_size=[640, 360], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.labels = labels
        self.anchors = anchors
        self.strides = strides
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.nms_option = nms_option
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        self.latest_detection = None
        
        # 跟踪目标ID，避免频繁切换目标
        self.tracked_person_id = None
        self.lost_count = 0  # 目标丢失计数

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [0, 0, 0])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            dets = aicube.anchorbasedet_post_process(
                results[0], results[1], results[2], 
                self.model_input_size, self.rgb888p_size, 
                self.strides, len(self.labels), 
                self.confidence_threshold, self.nms_threshold, 
                self.anchors, self.nms_option
            )
            self.latest_detection = dets
            return dets

    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            if dets:
                pl.osd_img.clear()
                # 绘制中心安全区域（可视化调试用）
                pl.osd_img.draw_rectangle(
                    center_x - center_width//2, 
                    center_y - center_height//2, 
                    center_width, 
                    center_height, 
                    color=(0, 255, 0, 100),  # 半透明绿色
                    thickness=2
                )
                
                for det_box in dets:
                    x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
                    w = float(x2 - x1) * self.display_size[0] // self.rgb888p_size[0]
                    h = float(y2 - y1) * self.display_size[1] // self.rgb888p_size[1]
                    x1 = int(x1 * self.display_size[0] // self.rgb888p_size[0])
                    y1 = int(y1 * self.display_size[1] // self.rgb888p_size[1])
                    x2 = int(x2 * self.display_size[0] // self.rgb888p_size[0])
                    y2 = int(y2 * self.display_size[1] // self.rgb888p_size[1])

                    # 放宽过滤条件，避免丢失边缘目标
                    if (h < (0.05 * self.display_size[0])):
                        continue
                    if (w < (0.1 * self.display_size[0]) and ((x1 < (0.05 * self.display_size[0])) or (x2 > (0.95 * self.display_size[0])))):
                        continue

                    # 绘制检测框和标签
                    pl.osd_img.draw_rectangle(x1, y1, int(w), int(h), color=(255, 0, 255, 0), thickness=2)
                    pl.osd_img.draw_string_advanced(x1, y1 - 50, 32, " " + self.labels[det_box[0]] + " " + str(round(det_box[1], 2)), color=(255, 0, 255, 0))
                    
                    # 发送协议数据
                    pto_data = pto.get_person_detect_data(x1, y1, w, h)
                    uart.send(pto_data)
            else:
                pl.osd_img.clear()
                # 绘制中心安全区域
                pl.osd_img.draw_rectangle(
                    center_x - center_width//2, 
                    center_y - center_height//2, 
                    center_width, 
                    center_height, 
                    color=(0, 255, 0, 100), 
                    thickness=2
                )

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        input_width = self.rgb888p_size[0]
        input_high = self.rgb888p_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        ratio = ratio_w if ratio_w < ratio_h else ratio_h
        
        new_w = int(ratio * input_width)
        new_h = int(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw - 0.1))
        
        return top, bottom, left, right

    def get_tracked_person(self):
        """优化目标跟踪：优先跟踪已锁定的目标，避免频繁切换"""
        global center_x, center_y
        
        if not self.latest_detection:
            self.lost_count += 1
            if self.lost_count > 8:
                self.tracked_person_id = None
            return None
            
        self.lost_count = 0  # 重置丢失计数
        person_boxes = [box for box in self.latest_detection if self.labels[int(box[0])] == "person"]
        
        if not person_boxes:
            return None
            
        # 如果已有跟踪目标，优先选择与上次位置最接近的目标
        if self.tracked_person_id is not None:
            # 计算与中心的距离，选择最近的
            def distance_to_center(box):
                _, _, x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                return abs(cx - center_x) + abs(cy - center_y)  # 曼哈顿距离
                
            # 选择离中心最近的目标
            return min(person_boxes, key=distance_to_center)
        else:
            # 没有跟踪目标时，选择最大的人体目标
            def box_area(box):
                x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
                return (x2 - x1) * (y2 - y1)
                
            # 锁定第一个目标
            largest_person = max(person_boxes, key=box_area)
            self.tracked_person_id = id(largest_person)  # 用内存地址作为临时ID
            return largest_person


def init_servos():
    """初始化舵机"""
    global pwm_lr, pwm_ud, last_lr_duty, last_ud_duty
    
    # 配置上下舵机
    pwm_io1 = FPIOA()
    pwm_io1.set_function(47, FPIOA.PWM3)
    pwm_ud = PWM(3)
    pwm_ud.freq(50)
    
    # 配置左右舵机
    pwm_io2 = FPIOA()
    pwm_io2.set_function(46, FPIOA.PWM2)
    pwm_lr = PWM(2)
    pwm_lr.freq(50)
    
    # 初始位置
    pwm_lr.duty_u16(mid_duty)
    pwm_ud.duty_u16(mid_duty)
    last_lr_duty = mid_duty
    last_ud_duty = mid_duty


def input_to_duty_cycle(input_min, input_max, input_value, last_duty):
    """将输入值映射到舵机占空比范围，并限制转动速度"""
    output_min = min_duty
    output_max = max_duty

    # 限制输入范围
    input_value = max(input_min, min(input_value, input_max))

    # 线性映射计算目标占空比
    slope = (output_max - output_min) / (input_max - input_min)
    target_duty = output_min + (input_value - input_min) * slope
    target_duty = int(max(output_min, min(target_duty, output_max)))

    # 单次转动步长
    max_step = 60  # 可根据舵机性能调整
    if target_duty > last_duty + max_step:
        target_duty = last_duty + max_step
    elif target_duty < last_duty - max_step:
        target_duty = last_duty - max_step

    return target_duty


def check_center_and_launch(person_det):
    """检查人体是否在中心区域足够长时间，触发发射"""
    global in_center_start_time, last_launch_time, gpio_objs
    
    person_box = person_det.get_tracked_person()
    current_time = utime.ticks_ms()
    
    if person_box:
        # 提取检测框中心
        _, _, x1, y1, x2, y2 = person_box
        center_x_obj = (x1 + x2) / 2
        center_y_obj = (y1 + y2) / 2
        
        # 计算相对于屏幕中心的偏移
        x_offset = center_x_obj - center_x
        y_offset = center_y_obj - center_y
        
        # 判断是否在中心安全区域内
        in_center_x = (abs(x_offset) <= center_width // 2)
        in_center_y = (abs(y_offset) <= center_height // 2)
        
        if in_center_x and in_center_y:
            # 人体在中心区域内
            if in_center_start_time is None:
                # 开始计时
                in_center_start_time = current_time
                print("目标已进入中心区域，开始计时...")
            else:
                # 计算已在中心区域的时间
                elapsed = utime.ticks_diff(current_time, in_center_start_time)
                if elapsed >= CENTER_DELAY:
                    # 检查是否达到发射间隔
                    if utime.ticks_diff(current_time, last_launch_time) >= LAUNCH_INTERVAL:
                        print(f"目标在中心区域停留满{CENTER_DELAY/1000}秒，触发发射！")
                        # 调用发射控制函数
                        control_gpios(gpio_objs)
                        last_launch_time = current_time
                    else:
                        remaining = LAUNCH_INTERVAL - utime.ticks_diff(current_time, last_launch_time)
                        print(f"距离下次发射还剩{remaining/1000:.1f}秒")
        else:
            # 人体离开中心区域，重置计时
            if in_center_start_time is not None:
                print("目标离开中心区域，重置计时")
                in_center_start_time = None
    else:
        # 未检测到人体，重置计时
        if in_center_start_time is not None:
            print("未检测到目标，重置计时")
            in_center_start_time = None


def control_servos_based_on_detection(person_det):
    """修正上下方向后的控制逻辑"""
    global pid_lr, pid_ud, kalman_lr, kalman_ud, last_lr_duty, last_ud_duty
    global center_x, center_y, center_width, center_height
    
    person_box = person_det.get_tracked_person()
    
    if person_box:
        # 提取检测框中心
        _, _, x1, y1, x2, y2 = person_box
        center_x_obj = (x1 + x2) / 2
        center_y_obj = (y1 + y2) / 2
        
        # 计算相对于屏幕中心的偏移
        x_offset = center_x_obj - center_x
        y_offset = center_y_obj - center_y
        
        # 判断是否在中心安全区域内
        in_center_x = (abs(x_offset) <= center_width // 2)
        in_center_y = (abs(y_offset) <= center_height // 2)
        
        if not in_center_x or not in_center_y:
            # 卡尔曼滤波平滑
            filtered_x = kalman_lr.update(x_offset)
            filtered_y = kalman_ud.update(y_offset)
            
            # PID计算
            pid_lr_value = pid_lr.pid_calc(0, filtered_x) if not in_center_x else 0
            pid_ud_value = pid_ud.pid_calc(0, filtered_y) if not in_center_y else 0
            
            # 方向控制：左右正确，上下反转
            duty_lr_value = input_to_duty_cycle(
                -(person_det.rgb888p_size[0] // 2), 
                (person_det.rgb888p_size[0] // 2), 
                pid_lr_value,  # 左右方向正确
                last_lr_duty
            )
            duty_ud_value = input_to_duty_cycle(
                -(person_det.rgb888p_size[1] // 2), 
                (person_det.rgb888p_size[1] // 2), 
                -pid_ud_value,  # 上下方向反转
                last_ud_duty
            )
            
            # 控制舵机
            if pwm_lr and pwm_ud:
                pwm_lr.duty_u16(duty_lr_value)
                pwm_ud.duty_u16(duty_ud_value)
                
            # 更新上次位置
            last_lr_duty = duty_lr_value
            last_ud_duty = duty_ud_value
            
        return True
    else:
        return False


def run_tracking_demo():
    """运行带发射功能的人体中心追踪演示"""
    global person_det, pl, gpio_objs
    
    try:
        # 1. 初始化GPIO（发射控制）
        gpio_objs = init_gpio_pins()
        print("GPIO初始化完成，准备就绪")
        
        # 2. 初始化舵机
        init_servos()
        print("舵机初始化完成，准备就绪")
        
        # 3. 配置参数
        rgb888p_size = [DETECT_WIDTH, DETECT_HEIGHT]
        display_size = [DETECT_WIDTH, DETECT_HEIGHT]
        display_mode = "lcd"
        
        # 4. 初始化PipeLine
        pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
        pl.create()
        
        # 5. 模型参数设置
        kmodel_path = "/sdcard/kmodel/person_detect_yolov5n.kmodel"
        confidence_threshold = 0.2
        nms_threshold = 0.6
        labels = ["person"]
        anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        
        # 6. 初始化人体检测实例
        person_det = PersonDetectionApp(
            kmodel_path, 
            model_input_size=[640, 640], 
            labels=labels, 
            anchors=anchors, 
            confidence_threshold=confidence_threshold, 
            nms_threshold=nms_threshold, 
            nms_option=False, 
            strides=[8, 16, 32], 
            rgb888p_size=rgb888p_size, 
            display_size=display_size, 
            debug_mode=0
        )
        person_det.config_preprocess()
        
        last_time = utime.ticks_ms()
        print("开始追踪...")
        while True:
            with ScopedTiming("total", 1):
                # 控制循环频率
                current_time = utime.ticks_ms()
                if utime.ticks_diff(current_time, last_time) < 20:
                    continue
                last_time = current_time
                
                # 获取当前帧并进行推理
                img = pl.get_frame()
                res = person_det.run(img)
                
                # 绘制结果（包含中心区域）
                person_det.draw_result(pl, res)
                pl.show_image()
                
                # 根据检测结果控制舵机
                control_servos_based_on_detection(person_det)
                
                # 检查是否满足发射条件
                check_center_and_launch(person_det)
                
                # 垃圾回收
                gc.collect()
                
    except Exception as e:
        print(f"人体追踪功能出错: {e}")
    finally:
        # 清理资源
        if person_det:
            person_det.deinit()
        if pl:
            pl.destroy()
        if pwm_lr:
            pwm_lr.deinit()
        if pwm_ud:
            pwm_ud.deinit()
        gc.collect()


if __name__ == "__main__":
    try:
        run_tracking_demo()
    except KeyboardInterrupt:
        print("用户终止程序")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
    