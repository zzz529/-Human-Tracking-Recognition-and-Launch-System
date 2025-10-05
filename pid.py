class PID:
    def __init__(self, kp, ki, kd, maxI, maxOut):
        #��̬����
        self.kp = kp
        self.ki = ki
        self.kd = kd
        #��̬����
        self.change_p = 0
        self.change_i = 0
        self.change_d = 0

        self.max_change_i = maxI    #�����޷�
        self.maxOutput = maxOut     #����޷�

        self.error_sum = 0  #��ǰ���
        self.last_error = 0 #֮ǰ���

    def change_zero(self):#PID�仯�ۼƵĲ�������
        self.change_p = 0
        self.change_i = 0
        self.change_d = 0

    def pid_calc(self, reference, feedback):#reference=Ŀ��λ��	feedback=��ǰλ��
        self.last_error = self.error_sum
        self.error_sum = reference - feedback #��ȡ�µ����

        #����΢��
        dout = (self.error_sum - self.last_error) * self.kd

        #�������
        pout = self.error_sum * self.kp

        #�������
        self.change_i += self.error_sum * self.ki

        #�����޷�
        if self.change_i > self.max_change_i :
            self.change_i = self.max_change_i
        elif self.change_i < -self.max_change_i:
            self.change_i = -self.max_change_i

        #�������
        self.output = pout + dout + self.change_i

        #����޷�
        if self.output > self.maxOutput:
            self.output =   self.maxOutput
        elif self.output < -self.maxOutput:
            self.output = -self.maxOutput

        return self.output
