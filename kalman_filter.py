import utime

class KalmanFilter:
    """
    卡尔曼滤波器实现，适用于一维运动目标跟踪
    状态变量: [位置, 速度]
    """
    def __init__(self, process_noise=0.1, measurement_noise=0.1, error_estimate=1.0):
        # 初始化状态估计 [位置, 速度]
        self.x = [0.0, 0.0]
        
        # 状态协方差矩阵，反映状态估计的不确定性
        self.P = [[error_estimate, 0],
                  [0, error_estimate]]
        
        # 状态转移矩阵
        self.F = [[1, 1],  # 位置 = 上一位置 + 速度 * 时间
                  [0, 1]]  # 速度保持不变
        
        # 测量矩阵，将状态映射到测量值
        self.H = [1, 0]  # 只能测量位置
        
        # 过程噪声协方差，反映系统模型的不确定性
        self.Q = [[process_noise, 0],
                  [0, process_noise]]
        
        # 测量噪声协方差，反映传感器的不确定性
        self.R = measurement_noise  # 测量噪声
        
        # 单位矩阵
        self.I = [[1, 0],
                  [0, 1]]
        
        self.last_time = utime.ticks_ms()

    def predict(self):
        """预测步骤：根据上一状态预测当前状态"""
        # 预测状态: x = F * x
        x0 = self.F[0][0] * self.x[0] + self.F[0][1] * self.x[1]
        x1 = self.F[1][0] * self.x[0] + self.F[1][1] * self.x[1]
        self.x = [x0, x1]
        
        # 预测协方差: P = F * P * F^T + Q
        p00 = self.F[0][0] * self.P[0][0] + self.F[0][1] * self.P[1][0]
        p01 = self.F[0][0] * self.P[0][1] + self.F[0][1] * self.P[1][1]
        p10 = self.F[1][0] * self.P[0][0] + self.F[1][1] * self.P[1][0]
        p11 = self.F[1][0] * self.P[0][1] + self.F[1][1] * self.P[1][1]
        
        # 添加过程噪声
        self.P[0][0] = p00 + self.Q[0][0]
        self.P[0][1] = p01 + self.Q[0][1]
        self.P[1][0] = p10 + self.Q[1][0]
        self.P[1][1] = p11 + self.Q[1][1]

    def update(self, measurement):
        """
        更新步骤：根据测量值修正预测值
        参数: measurement - 传感器测量值
        返回: 滤波后的估计值
        """
        # 计算时间间隔（秒）
        current_time = utime.ticks_ms()
        dt = utime.ticks_diff(current_time, self.last_time) / 1000.0
        self.last_time = current_time
        
        # 更新状态转移矩阵中的时间项
        self.F[0][1] = dt
        
        # 先执行预测
        self.predict()
        
        # 计算卡尔曼增益
        # K = P * H^T / (H * P * H^T + R)
        denominator = self.H[0] * self.P[0][0] + self.H[1] * self.P[1][0] + self.R
        k0 = (self.P[0][0] * self.H[0] + self.P[0][1] * self.H[1]) / denominator
        k1 = (self.P[1][0] * self.H[0] + self.P[1][1] * self.H[1]) / denominator
        
        # 更新状态估计
        # x = x + K * (z - H * x)
        innovation = measurement - (self.H[0] * self.x[0] + self.H[1] * self.x[1])
        self.x[0] += k0 * innovation
        self.x[1] += k1 * innovation
        
        # 更新状态协方差矩阵
        # P = (I - K * H) * P
        p00 = (self.I[0][0] - k0 * self.H[0]) * self.P[0][0] + (self.I[0][1] - k0 * self.H[1]) * self.P[1][0]
        p01 = (self.I[0][0] - k0 * self.H[0]) * self.P[0][1] + (self.I[0][1] - k0 * self.H[1]) * self.P[1][1]
        p10 = (self.I[1][0] - k1 * self.H[0]) * self.P[0][0] + (self.I[1][1] - k1 * self.H[1]) * self.P[1][0]
        p11 = (self.I[1][0] - k1 * self.H[0]) * self.P[0][1] + (self.I[1][1] - k1 * self.H[1]) * self.P[1][1]
        
        self.P = [[p00, p01], [p10, p11]]
        
        # 返回滤波后的位置估计
        return self.x[0]
    
    def get_velocity(self):
        """返回估计的速度"""
        return self.x[1]
    
    def reset(self):
        """重置滤波器状态"""
        self.x = [0.0, 0.0]
        self.last_time = utime.ticks_ms()
