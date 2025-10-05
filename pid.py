class PID:
    def __init__(self, kp, ki, kd, maxI, maxOut):
        #静态参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        #动态参数
        self.change_p = 0
        self.change_i = 0
        self.change_d = 0

        self.max_change_i = maxI    #积分限幅
        self.maxOutput = maxOut     #输出限幅

        self.error_sum = 0  #当前误差
        self.last_error = 0 #之前误差

    def change_zero(self):#PID变化累计的参数清零
        self.change_p = 0
        self.change_i = 0
        self.change_d = 0

    def pid_calc(self, reference, feedback):#reference=目标位置	feedback=当前位置
        self.last_error = self.error_sum
        self.error_sum = reference - feedback #获取新的误差

        #计算微分
        dout = (self.error_sum - self.last_error) * self.kd

        #计算比例
        pout = self.error_sum * self.kp

        #计算积分
        self.change_i += self.error_sum * self.ki

        #积分限幅
        if self.change_i > self.max_change_i :
            self.change_i = self.max_change_i
        elif self.change_i < -self.max_change_i:
            self.change_i = -self.max_change_i

        #计算输出
        self.output = pout + dout + self.change_i

        #输出限幅
        if self.output > self.maxOutput:
            self.output =   self.maxOutput
        elif self.output < -self.maxOutput:
            self.output = -self.maxOutput

        return self.output
