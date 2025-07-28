import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
import pickle
import gc
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class PredictionModels:
    def __init__(self, data_loader, models_dir: str = "saved_models"):
        """
        初始化预测模型类
        :param data_loader: 数据加载器实例
        :param models_dir: 保存模型的目录
        """
        self.data_loader = data_loader
        self.models_dir = models_dir
        self.arima_models = {}  # 存储ARIMA模型 {vm_id: model}
        self.lstm_models = {}   # 存储LSTM模型 {vm_id: model}
        self.scalers = {}       # 存储标准化器 {vm_id: scaler}
        
        # 创建模型保存目录
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, 'arima'), exist_ok=True)
        os.makedirs(os.path.join(models_dir, 'lstm'), exist_ok=True)
        os.makedirs(os.path.join(models_dir, 'scalers'), exist_ok=True)
        
        # 设置序列长度和预测长度
        self.sequence_length = 5  # 使用5个时间窗口的数据进行预测
        self.prediction_steps = 3  # 预测未来3个时间窗口的数据
        
    def prepare_lstm_data(self, vm_id: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        准备LSTM模型的训练数据
        :param vm_id: 虚拟机ID
        :return: X训练数据, Y训练数据, 标准化器
        """
        # 获取指定虚拟机的所有时间窗口数据
        vm_data = self.data_loader.time_windows.get(vm_id, {})
        if not vm_data:
            raise ValueError(f"没有找到虚拟机 {vm_id} 的数据")
        
        # 获取窗口总数
        total_windows = max(vm_data.keys()) + 1
        
        # 提取内存使用率数据
        memory_usage = [vm_data.get(i, {'memory_usage': 0})['memory_usage'] for i in range(total_windows)]
        memory_usage = np.array(memory_usage).reshape(-1, 1)
        
        # 标准化数据
        scaler = MinMaxScaler(feature_range=(0, 1))
        memory_usage_scaled = scaler.fit_transform(memory_usage)
        
        # 创建输入序列和目标值
        X, y = [], []
        for i in range(len(memory_usage_scaled) - self.sequence_length - self.prediction_steps + 1):
            X.append(memory_usage_scaled[i:i+self.sequence_length])
            y.append(memory_usage_scaled[i+self.sequence_length:i+self.sequence_length+self.prediction_steps])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, scaler
    
    def build_lstm_model(self, input_shape: Tuple) -> Sequential:
        """
        构建LSTM模型
        :param input_shape: 输入形状
        :return: LSTM模型
        """
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(self.prediction_steps))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_lstm_model(self, vm_id: int, force_retrain: bool = False) -> Sequential:
        """
        训练LSTM模型
        :param vm_id: 虚拟机ID
        :param force_retrain: 是否强制重新训练
        :return: 训练好的LSTM模型
        """
        model_path = os.path.join(self.models_dir, 'lstm', f'lstm_vm_{vm_id}.h5')
        scaler_path = os.path.join(self.models_dir, 'scalers', f'scaler_vm_{vm_id}.pkl')
        
        # 检查是否已有训练好的模型
        if os.path.exists(model_path) and os.path.exists(scaler_path) and not force_retrain:
            print(f"加载虚拟机 {vm_id} 的LSTM模型")
            model = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            self.lstm_models[vm_id] = model
            self.scalers[vm_id] = scaler
            return model
        
        print(f"训练虚拟机 {vm_id} 的LSTM模型")
        # 准备训练数据
        X, y, scaler = self.prepare_lstm_data(vm_id)
        
        if len(X) == 0:
            raise ValueError(f"虚拟机 {vm_id} 的训练数据为空")
        
        # 构建并训练模型
        model = self.build_lstm_model((X.shape[1], X.shape[2]))
        early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.0001)
        model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
        
        # 保存模型和标准化器
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # 存储模型和标准化器
        self.lstm_models[vm_id] = model
        self.scalers[vm_id] = scaler
        
        # 清理内存
        tf.keras.backend.clear_session()
        gc.collect()
        
        return model
    
    def prepare_arima_data(self, vm_id: int) -> np.ndarray:
        """
        准备ARIMA模型的训练数据
        :param vm_id: 虚拟机ID
        :return: CPU使用率数据数组
        """
        # 获取指定虚拟机的所有时间窗口数据
        vm_data = self.data_loader.time_windows.get(vm_id, {})
        if not vm_data:
            raise ValueError(f"没有找到虚拟机 {vm_id} 的数据")
        
        # 获取窗口总数
        total_windows = max(vm_data.keys()) + 1
        
        # 提取CPU使用率数据
        cpu_usage = [vm_data.get(i, {'cpu_usage': 0})['cpu_usage'] for i in range(total_windows)]
        return np.array(cpu_usage)
    
    def train_arima_model(self, vm_id: int, force_retrain: bool = False) -> auto_arima:
        """
        训练ARIMA模型
        :param vm_id: 虚拟机ID
        :param force_retrain: 是否强制重新训练
        :return: 训练好的ARIMA模型
        """
        model_path = os.path.join(self.models_dir, 'arima', f'arima_vm_{vm_id}.pkl')
        
        # 检查是否已有训练好的模型
        if os.path.exists(model_path) and not force_retrain:
            print(f"加载虚拟机 {vm_id} 的ARIMA模型")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.arima_models[vm_id] = model
            return model
        
        print(f"训练虚拟机 {vm_id} 的ARIMA模型")
        # 准备训练数据
        cpu_usage = self.prepare_arima_data(vm_id)
        
        # 训练ARIMA模型
        model = auto_arima(
            cpu_usage,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            m=1,
            d=0,
            seasonal=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_jobs=-1
        )
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 存储模型
        self.arima_models[vm_id] = model
        
        # 清理内存
        gc.collect()
        
        return model
    
    def predict_cpu_usage(self, vm_id: int, current_window: int) -> List[float]:
        """
        预测CPU使用率
        :param vm_id: 虚拟机ID
        :param current_window: 当前时间窗口
        :return: 预测的CPU使用率列表
        """
        if vm_id not in self.arima_models:
            raise ValueError(f"虚拟机 {vm_id} 的ARIMA模型未加载")
        
        # 获取模型
        model = self.arima_models[vm_id]
        
        # 预测未来使用率
        forecasts = model.predict(n_periods=self.prediction_steps)
        
        # 确保预测结果在合理范围内
        forecasts = np.clip(forecasts, 0, 100)
        
        return forecasts.tolist()
    
    def predict_memory_usage(self, vm_id: int, current_window: int) -> List[float]:
        """
        预测内存使用率
        :param vm_id: 虚拟机ID
        :param current_window: 当前时间窗口
        :return: 预测的内存使用率列表
        """
        if vm_id not in self.lstm_models or vm_id not in self.scalers:
            raise ValueError(f"虚拟机 {vm_id} 的LSTM模型或标准化器未加载")
        
        # 获取模型和标准化器
        model = self.lstm_models[vm_id]
        scaler = self.scalers[vm_id]
        
        # 获取最近的序列数据
        start_window = max(0, current_window - self.sequence_length + 1)
        sequence_data = self.data_loader.get_vm_data_sequence(vm_id, start_window, current_window)
        
        # 提取内存使用率并填充零值（如果序列不够长）
        memory_usage = [d['memory_usage'] for d in sequence_data]
        memory_usage = memory_usage[-self.sequence_length:] if len(memory_usage) >= self.sequence_length else ([0] * (self.sequence_length - len(memory_usage))) + memory_usage
        
        # 转换为NumPy数组并重塑
        memory_usage = np.array(memory_usage).reshape(-1, 1)
        
        # 标准化数据
        memory_usage_scaled = scaler.transform(memory_usage)
        
        # 重塑为LSTM输入格式
        X = memory_usage_scaled.reshape(1, self.sequence_length, 1)
        
        # 预测未来使用率
        predictions_scaled = model.predict(X)
        
        # 反向标准化
        predicted_memory_usage = scaler.inverse_transform(predictions_scaled.reshape(self.prediction_steps, 1))
        
        # 确保预测结果在合理范围内
        predicted_memory_usage = np.clip(predicted_memory_usage, 0, 100)
        
        return predicted_memory_usage.flatten().tolist()
    
    def predict_future_data(self, current_window: int) -> Dict[int, List[Dict]]:
        """
        预测所有虚拟机未来的使用数据
        :param current_window: 当前时间窗口
        :return: 预测的未来数据 {vm_id: [{cpu_usage, memory_usage, risk_level}, ...]}
        """
        predictions = {}
        
        # 对每个虚拟机进行预测
        for vm_id in self.data_loader.vm_instances.keys():
            cpu_predictions = self.predict_cpu_usage(vm_id, current_window)
            memory_predictions = self.predict_memory_usage(vm_id, current_window)
            
            # 获取最近的风险评级
            risk_level = self.data_loader.time_windows[vm_id][current_window]['risk_level']
            
            # 创建预测数据序列
            vm_predictions = []
            for i in range(self.prediction_steps):
                vm_predictions.append({
                    'cpu_usage': cpu_predictions[i],
                    'memory_usage': memory_predictions[i],
                    'risk_level': risk_level  # 假设风险评级在预测期间保持不变
                })
            
            predictions[vm_id] = vm_predictions
        
        return predictions
    
    def train_all_models(self, force_retrain: bool = False):
        """
        为所有虚拟机训练模型
        :param force_retrain: 是否强制重新训练所有模型
        """
        for vm_id in self.data_loader.vm_instances.keys():
            try:
                self.train_arima_model(vm_id, force_retrain)
                self.train_lstm_model(vm_id, force_retrain)
            except Exception as e:
                print(f"训练虚拟机 {vm_id} 的模型时出错: {str(e)}")
                continue 