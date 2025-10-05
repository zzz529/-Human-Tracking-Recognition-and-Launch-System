from machine import FPIOA, Pin
import time

# ��ʼ���������ã�ȫ��ִ��һ�μ��ɣ�
def init_gpio_pins():
    """��ʼ��32�ź�42������ΪGPIO����"""
    fpioa = FPIOA()
    
    # 32����������
    physical_pin32 = 32
    target_gpio32 = FPIOA.GPIO32
    current_func32 = fpioa.get_pin_func(physical_pin32)
    if current_func32 != target_gpio32:
        fpioa.set_function(physical_pin32, target_gpio32)
        print(f"32������������Ϊ{target_gpio32}")
    else:
        print("32�������Ѿ���")
    
    # 42����������
    physical_pin42 = 42
    target_gpio42 = FPIOA.GPIO42
    current_func42 = fpioa.get_pin_func(physical_pin42)
    if current_func42 != target_gpio42:
        fpioa.set_function(physical_pin42, target_gpio42)
        print(f"42������������Ϊ{target_gpio42}")
    else:
        print("42�������Ѿ���")
    
    # ���س�ʼ��������Ŷ���
    return {
        "gpio32": Pin(target_gpio32, Pin.OUT, value=0),
        "gpio42": Pin(target_gpio42, Pin.OUT, value=1)  # 42��Ĭ�ϸߵ�ƽ
    }

def control_gpios(gpio_objs):
    """
    ����32���������0.1��ߵ�ƽ��رգ�42�����ű��ָߵ�ƽ
    
    ����:
        gpio_objs: ��init_gpio_pins()���ص����Ŷ����ֵ�
    """
    try:
        # 32����������ߵ�ƽ
        gpio_objs["gpio32"].value(1)
        print("32����������ߵ�ƽ")
        
        # ����0.1���ر�
        time.sleep(0.1)
        gpio_objs["gpio32"].value(0)
        print("32�������ѹرգ��͵�ƽ��")
        
        print("������ɣ�42�����ű��ָߵ�ƽ")
        
    except KeyboardInterrupt:
        # �жϴ���
        gpio_objs["gpio32"].value(0)
        gpio_objs["gpio42"].value(0)
        print("�����жϣ������������õ�")