from machine import FPIOA, Pin
import time

# 初始化引脚配置（全局执行一次即可）
def init_gpio_pins():
    """初始化32号和42号引脚为GPIO功能"""
    fpioa = FPIOA()
    
    # 32号引脚配置
    physical_pin32 = 32
    target_gpio32 = FPIOA.GPIO32
    current_func32 = fpioa.get_pin_func(physical_pin32)
    if current_func32 != target_gpio32:
        fpioa.set_function(physical_pin32, target_gpio32)
        print(f"32号引脚已配置为{target_gpio32}")
    else:
        print("32号引脚已就绪")
    
    # 42号引脚配置
    physical_pin42 = 42
    target_gpio42 = FPIOA.GPIO42
    current_func42 = fpioa.get_pin_func(physical_pin42)
    if current_func42 != target_gpio42:
        fpioa.set_function(physical_pin42, target_gpio42)
        print(f"42号引脚已配置为{target_gpio42}")
    else:
        print("42号引脚已就绪")
    
    # 返回初始化后的引脚对象
    return {
        "gpio32": Pin(target_gpio32, Pin.OUT, value=0),
        "gpio42": Pin(target_gpio42, Pin.OUT, value=1)  # 42号默认高电平
    }

def control_gpios(gpio_objs):
    """
    控制32号引脚输出0.1秒高电平后关闭，42号引脚保持高电平
    
    参数:
        gpio_objs: 由init_gpio_pins()返回的引脚对象字典
    """
    try:
        # 32号引脚输出高电平
        gpio_objs["gpio32"].value(1)
        print("32号引脚输出高电平")
        
        # 保持0.1秒后关闭
        time.sleep(0.1)
        gpio_objs["gpio32"].value(0)
        print("32号引脚已关闭（低电平）")
        
        print("操作完成，42号引脚保持高电平")
        
    except KeyboardInterrupt:
        # 中断处理
        gpio_objs["gpio32"].value(0)
        gpio_objs["gpio42"].value(0)
        print("程序被中断，所有引脚已置低")