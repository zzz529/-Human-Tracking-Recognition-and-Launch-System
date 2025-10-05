import utime

class KalmanFilter:
    """
    �������˲���ʵ�֣�������һά�˶�Ŀ�����
    ״̬����: [λ��, �ٶ�]
    """
    def __init__(self, process_noise=0.1, measurement_noise=0.1, error_estimate=1.0):
        # ��ʼ��״̬���� [λ��, �ٶ�]
        self.x = [0.0, 0.0]
        
        # ״̬Э������󣬷�ӳ״̬���ƵĲ�ȷ����
        self.P = [[error_estimate, 0],
                  [0, error_estimate]]
        
        # ״̬ת�ƾ���
        self.F = [[1, 1],  # λ�� = ��һλ�� + �ٶ� * ʱ��
                  [0, 1]]  # �ٶȱ��ֲ���
        
        # �������󣬽�״̬ӳ�䵽����ֵ
        self.H = [1, 0]  # ֻ�ܲ���λ��
        
        # ��������Э�����ӳϵͳģ�͵Ĳ�ȷ����
        self.Q = [[process_noise, 0],
                  [0, process_noise]]
        
        # ��������Э�����ӳ�������Ĳ�ȷ����
        self.R = measurement_noise  # ��������
        
        # ��λ����
        self.I = [[1, 0],
                  [0, 1]]
        
        self.last_time = utime.ticks_ms()

    def predict(self):
        """Ԥ�ⲽ�裺������һ״̬Ԥ�⵱ǰ״̬"""
        # Ԥ��״̬: x = F * x
        x0 = self.F[0][0] * self.x[0] + self.F[0][1] * self.x[1]
        x1 = self.F[1][0] * self.x[0] + self.F[1][1] * self.x[1]
        self.x = [x0, x1]
        
        # Ԥ��Э����: P = F * P * F^T + Q
        p00 = self.F[0][0] * self.P[0][0] + self.F[0][1] * self.P[1][0]
        p01 = self.F[0][0] * self.P[0][1] + self.F[0][1] * self.P[1][1]
        p10 = self.F[1][0] * self.P[0][0] + self.F[1][1] * self.P[1][0]
        p11 = self.F[1][0] * self.P[0][1] + self.F[1][1] * self.P[1][1]
        
        # ��ӹ�������
        self.P[0][0] = p00 + self.Q[0][0]
        self.P[0][1] = p01 + self.Q[0][1]
        self.P[1][0] = p10 + self.Q[1][0]
        self.P[1][1] = p11 + self.Q[1][1]

    def update(self, measurement):
        """
        ���²��裺���ݲ���ֵ����Ԥ��ֵ
        ����: measurement - ����������ֵ
        ����: �˲���Ĺ���ֵ
        """
        # ����ʱ�������룩
        current_time = utime.ticks_ms()
        dt = utime.ticks_diff(current_time, self.last_time) / 1000.0
        self.last_time = current_time
        
        # ����״̬ת�ƾ����е�ʱ����
        self.F[0][1] = dt
        
        # ��ִ��Ԥ��
        self.predict()
        
        # ���㿨��������
        # K = P * H^T / (H * P * H^T + R)
        denominator = self.H[0] * self.P[0][0] + self.H[1] * self.P[1][0] + self.R
        k0 = (self.P[0][0] * self.H[0] + self.P[0][1] * self.H[1]) / denominator
        k1 = (self.P[1][0] * self.H[0] + self.P[1][1] * self.H[1]) / denominator
        
        # ����״̬����
        # x = x + K * (z - H * x)
        innovation = measurement - (self.H[0] * self.x[0] + self.H[1] * self.x[1])
        self.x[0] += k0 * innovation
        self.x[1] += k1 * innovation
        
        # ����״̬Э�������
        # P = (I - K * H) * P
        p00 = (self.I[0][0] - k0 * self.H[0]) * self.P[0][0] + (self.I[0][1] - k0 * self.H[1]) * self.P[1][0]
        p01 = (self.I[0][0] - k0 * self.H[0]) * self.P[0][1] + (self.I[0][1] - k0 * self.H[1]) * self.P[1][1]
        p10 = (self.I[1][0] - k1 * self.H[0]) * self.P[0][0] + (self.I[1][1] - k1 * self.H[1]) * self.P[1][0]
        p11 = (self.I[1][0] - k1 * self.H[0]) * self.P[0][1] + (self.I[1][1] - k1 * self.H[1]) * self.P[1][1]
        
        self.P = [[p00, p01], [p10, p11]]
        
        # �����˲����λ�ù���
        return self.x[0]
    
    def get_velocity(self):
        """���ع��Ƶ��ٶ�"""
        return self.x[1]
    
    def reset(self):
        """�����˲���״̬"""
        self.x = [0.0, 0.0]
        self.last_time = utime.ticks_ms()
