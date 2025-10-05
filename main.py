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

# ����GPIO����ģ��
from gpio_control import init_gpio_pins, control_gpios

# ȫ�ֱ���
person_det = None
pl = None
sensor = None
gpio_objs = None  # GPIO���ƶ���

# ���������ر���
in_center_start_time = None  # ��ʼλ�����������ʱ��
CENTER_DELAY = 3000  # ��������ͣ��ʱ����ֵ�����룩
last_launch_time = 0  # �ϴη���ʱ�䣬����Ƶ������
LAUNCH_INTERVAL = 5000  # �����������룩

# ͨ��Э�����
from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart
uart = YbUart(baudrate=115200)
pto = YbProtocol()

###############################�������#####################################################
min_duty = 1638      # 2.5% ռ�ձ�
max_duty = 8192      # 12.5% ռ�ձ�
mid_duty = 4915      # 7.5% ռ�ձ� (�м�λ��)
pwm_lr = None        # ���Ҷ��
pwm_ud = None        # ���¶��

# ��¼����ϴ�λ�ã���������ת���ٶ�
last_lr_duty = mid_duty
last_ud_duty = mid_duty

###############################�������#####################################################
DETECT_WIDTH = 640
DETECT_HEIGHT = 480

# ������������
CENTER_REGION_RATIO = 0.2  
center_width = int(DETECT_WIDTH * CENTER_REGION_RATIO)
center_height = int(DETECT_HEIGHT * CENTER_REGION_RATIO)
center_x = DETECT_WIDTH // 2
center_y = DETECT_HEIGHT // 2

###############################PID����######################################
lr_kp = 0.12    # ���ұ���ϵ��
lr_ki = 0.01    # ���һ���ϵ��
lr_kd = 0.001   # ����΢��ϵ��
lr_max_out = 100  # ����������

ud_kp = 0.12    # ���±���ϵ��
ud_ki = 0.01    # ���»���ϵ��
ud_kd = 0.001   # ����΢��ϵ��
ud_max_out = 100  # ����������

pid_lr = PID(lr_kp, lr_ki, lr_kd, lr_max_out, lr_max_out)
pid_ud = PID(ud_kp, ud_ki, ud_kd, ud_max_out, ud_max_out)

###############################�������˲���######################################
kalman_lr = KalmanFilter(process_noise=0.1, measurement_noise=0.6, error_estimate=1.0)
kalman_ud = KalmanFilter(process_noise=0.1, measurement_noise=0.6, error_estimate=1.0)

# �Զ�����������
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
        
        # ����Ŀ��ID������Ƶ���л�Ŀ��
        self.tracked_person_id = None
        self.lost_count = 0  # Ŀ�궪ʧ����

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
                # �������İ�ȫ���򣨿��ӻ������ã�
                pl.osd_img.draw_rectangle(
                    center_x - center_width//2, 
                    center_y - center_height//2, 
                    center_width, 
                    center_height, 
                    color=(0, 255, 0, 100),  # ��͸����ɫ
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

                    # �ſ�������������ⶪʧ��ԵĿ��
                    if (h < (0.05 * self.display_size[0])):
                        continue
                    if (w < (0.1 * self.display_size[0]) and ((x1 < (0.05 * self.display_size[0])) or (x2 > (0.95 * self.display_size[0])))):
                        continue

                    # ���Ƽ���ͱ�ǩ
                    pl.osd_img.draw_rectangle(x1, y1, int(w), int(h), color=(255, 0, 255, 0), thickness=2)
                    pl.osd_img.draw_string_advanced(x1, y1 - 50, 32, " " + self.labels[det_box[0]] + " " + str(round(det_box[1], 2)), color=(255, 0, 255, 0))
                    
                    # ����Э������
                    pto_data = pto.get_person_detect_data(x1, y1, w, h)
                    uart.send(pto_data)
            else:
                pl.osd_img.clear()
                # �������İ�ȫ����
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
        """�Ż�Ŀ����٣����ȸ�����������Ŀ�꣬����Ƶ���л�"""
        global center_x, center_y
        
        if not self.latest_detection:
            self.lost_count += 1
            if self.lost_count > 8:
                self.tracked_person_id = None
            return None
            
        self.lost_count = 0  # ���ö�ʧ����
        person_boxes = [box for box in self.latest_detection if self.labels[int(box[0])] == "person"]
        
        if not person_boxes:
            return None
            
        # ������и���Ŀ�꣬����ѡ�����ϴ�λ����ӽ���Ŀ��
        if self.tracked_person_id is not None:
            # ���������ĵľ��룬ѡ�������
            def distance_to_center(box):
                _, _, x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                return abs(cx - center_x) + abs(cy - center_y)  # �����پ���
                
            # ѡ�������������Ŀ��
            return min(person_boxes, key=distance_to_center)
        else:
            # û�и���Ŀ��ʱ��ѡ����������Ŀ��
            def box_area(box):
                x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
                return (x2 - x1) * (y2 - y1)
                
            # ������һ��Ŀ��
            largest_person = max(person_boxes, key=box_area)
            self.tracked_person_id = id(largest_person)  # ���ڴ��ַ��Ϊ��ʱID
            return largest_person


def init_servos():
    """��ʼ�����"""
    global pwm_lr, pwm_ud, last_lr_duty, last_ud_duty
    
    # �������¶��
    pwm_io1 = FPIOA()
    pwm_io1.set_function(47, FPIOA.PWM3)
    pwm_ud = PWM(3)
    pwm_ud.freq(50)
    
    # �������Ҷ��
    pwm_io2 = FPIOA()
    pwm_io2.set_function(46, FPIOA.PWM2)
    pwm_lr = PWM(2)
    pwm_lr.freq(50)
    
    # ��ʼλ��
    pwm_lr.duty_u16(mid_duty)
    pwm_ud.duty_u16(mid_duty)
    last_lr_duty = mid_duty
    last_ud_duty = mid_duty


def input_to_duty_cycle(input_min, input_max, input_value, last_duty):
    """������ֵӳ�䵽���ռ�ձȷ�Χ��������ת���ٶ�"""
    output_min = min_duty
    output_max = max_duty

    # �������뷶Χ
    input_value = max(input_min, min(input_value, input_max))

    # ����ӳ�����Ŀ��ռ�ձ�
    slope = (output_max - output_min) / (input_max - input_min)
    target_duty = output_min + (input_value - input_min) * slope
    target_duty = int(max(output_min, min(target_duty, output_max)))

    # ����ת������
    max_step = 60  # �ɸ��ݶ�����ܵ���
    if target_duty > last_duty + max_step:
        target_duty = last_duty + max_step
    elif target_duty < last_duty - max_step:
        target_duty = last_duty - max_step

    return target_duty


def check_center_and_launch(person_det):
    """��������Ƿ������������㹻��ʱ�䣬��������"""
    global in_center_start_time, last_launch_time, gpio_objs
    
    person_box = person_det.get_tracked_person()
    current_time = utime.ticks_ms()
    
    if person_box:
        # ��ȡ��������
        _, _, x1, y1, x2, y2 = person_box
        center_x_obj = (x1 + x2) / 2
        center_y_obj = (y1 + y2) / 2
        
        # �����������Ļ���ĵ�ƫ��
        x_offset = center_x_obj - center_x
        y_offset = center_y_obj - center_y
        
        # �ж��Ƿ������İ�ȫ������
        in_center_x = (abs(x_offset) <= center_width // 2)
        in_center_y = (abs(y_offset) <= center_height // 2)
        
        if in_center_x and in_center_y:
            # ����������������
            if in_center_start_time is None:
                # ��ʼ��ʱ
                in_center_start_time = current_time
                print("Ŀ���ѽ����������򣬿�ʼ��ʱ...")
            else:
                # �����������������ʱ��
                elapsed = utime.ticks_diff(current_time, in_center_start_time)
                if elapsed >= CENTER_DELAY:
                    # ����Ƿ�ﵽ������
                    if utime.ticks_diff(current_time, last_launch_time) >= LAUNCH_INTERVAL:
                        print(f"Ŀ������������ͣ����{CENTER_DELAY/1000}�룬�������䣡")
                        # ���÷�����ƺ���
                        control_gpios(gpio_objs)
                        last_launch_time = current_time
                    else:
                        remaining = LAUNCH_INTERVAL - utime.ticks_diff(current_time, last_launch_time)
                        print(f"�����´η��仹ʣ{remaining/1000:.1f}��")
        else:
            # �����뿪�����������ü�ʱ
            if in_center_start_time is not None:
                print("Ŀ���뿪�����������ü�ʱ")
                in_center_start_time = None
    else:
        # δ��⵽���壬���ü�ʱ
        if in_center_start_time is not None:
            print("δ��⵽Ŀ�꣬���ü�ʱ")
            in_center_start_time = None


def control_servos_based_on_detection(person_det):
    """�������·����Ŀ����߼�"""
    global pid_lr, pid_ud, kalman_lr, kalman_ud, last_lr_duty, last_ud_duty
    global center_x, center_y, center_width, center_height
    
    person_box = person_det.get_tracked_person()
    
    if person_box:
        # ��ȡ��������
        _, _, x1, y1, x2, y2 = person_box
        center_x_obj = (x1 + x2) / 2
        center_y_obj = (y1 + y2) / 2
        
        # �����������Ļ���ĵ�ƫ��
        x_offset = center_x_obj - center_x
        y_offset = center_y_obj - center_y
        
        # �ж��Ƿ������İ�ȫ������
        in_center_x = (abs(x_offset) <= center_width // 2)
        in_center_y = (abs(y_offset) <= center_height // 2)
        
        if not in_center_x or not in_center_y:
            # �������˲�ƽ��
            filtered_x = kalman_lr.update(x_offset)
            filtered_y = kalman_ud.update(y_offset)
            
            # PID����
            pid_lr_value = pid_lr.pid_calc(0, filtered_x) if not in_center_x else 0
            pid_ud_value = pid_ud.pid_calc(0, filtered_y) if not in_center_y else 0
            
            # ������ƣ�������ȷ�����·�ת
            duty_lr_value = input_to_duty_cycle(
                -(person_det.rgb888p_size[0] // 2), 
                (person_det.rgb888p_size[0] // 2), 
                pid_lr_value,  # ���ҷ�����ȷ
                last_lr_duty
            )
            duty_ud_value = input_to_duty_cycle(
                -(person_det.rgb888p_size[1] // 2), 
                (person_det.rgb888p_size[1] // 2), 
                -pid_ud_value,  # ���·���ת
                last_ud_duty
            )
            
            # ���ƶ��
            if pwm_lr and pwm_ud:
                pwm_lr.duty_u16(duty_lr_value)
                pwm_ud.duty_u16(duty_ud_value)
                
            # �����ϴ�λ��
            last_lr_duty = duty_lr_value
            last_ud_duty = duty_ud_value
            
        return True
    else:
        return False


def run_tracking_demo():
    """���д����书�ܵ���������׷����ʾ"""
    global person_det, pl, gpio_objs
    
    try:
        # 1. ��ʼ��GPIO��������ƣ�
        gpio_objs = init_gpio_pins()
        print("GPIO��ʼ����ɣ�׼������")
        
        # 2. ��ʼ�����
        init_servos()
        print("�����ʼ����ɣ�׼������")
        
        # 3. ���ò���
        rgb888p_size = [DETECT_WIDTH, DETECT_HEIGHT]
        display_size = [DETECT_WIDTH, DETECT_HEIGHT]
        display_mode = "lcd"
        
        # 4. ��ʼ��PipeLine
        pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
        pl.create()
        
        # 5. ģ�Ͳ�������
        kmodel_path = "/sdcard/kmodel/person_detect_yolov5n.kmodel"
        confidence_threshold = 0.2
        nms_threshold = 0.6
        labels = ["person"]
        anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        
        # 6. ��ʼ��������ʵ��
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
        print("��ʼ׷��...")
        while True:
            with ScopedTiming("total", 1):
                # ����ѭ��Ƶ��
                current_time = utime.ticks_ms()
                if utime.ticks_diff(current_time, last_time) < 20:
                    continue
                last_time = current_time
                
                # ��ȡ��ǰ֡����������
                img = pl.get_frame()
                res = person_det.run(img)
                
                # ���ƽ����������������
                person_det.draw_result(pl, res)
                pl.show_image()
                
                # ���ݼ�������ƶ��
                control_servos_based_on_detection(person_det)
                
                # ����Ƿ����㷢������
                check_center_and_launch(person_det)
                
                # ��������
                gc.collect()
                
    except Exception as e:
        print(f"����׷�ٹ��ܳ���: {e}")
    finally:
        # ������Դ
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
        print("�û���ֹ����")
    except Exception as e:
        print(f"�����쳣: {e}")
    finally:
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
    