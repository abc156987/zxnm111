"""
智能问答系统（总体设计 + 文本分类）
整合版 - 可直接运行
作者：编程导师
功能：
1. 智能问答（基于豆包API）
2. 文本分类（基于TensorFlow模型）
3. Flask Web服务
"""

import os
import sys
import json
import socket
import numpy as np
import http.client
from typing import Dict, List, Optional, Any

# Flask 相关导入
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# TensorFlow/文本分类相关导入
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# ======================== 全局配置 ========================
# Flask 配置
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# 豆包 API 配置
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_API_KEY = "48a29225-a258-471c-97e6-4e1ebef8ae35"  # 请替换为你的实际API Key
DOUBAO_MODEL = "doubao-seed-1-6-250615"

# 文本分类配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_CLASSIFY_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'my_model.h5')
TEXT_CLASSIFY_VOCAB_PATH = os.path.join(BASE_DIR, 'data', 'cnews.vocab.txt')
TEXT_CLASSIFY_CATEGORIES = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
TEXT_CLASSIFY_SEQ_LENGTH = 600

# 禁用GPU（避免环境依赖问题）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# ======================== 豆包API类 ========================
class DoubaoAPI:
    """豆包API封装类（智能问答核心）"""
    def __init__(self):
        self.api_url = DOUBAO_API_URL
        self.api_key = DOUBAO_API_KEY
        self.model = DOUBAO_MODEL
        
        # 解析URL
        if "https://" in self.api_url:
            self.host = self.api_url.replace("https://", "").split("/")[0]
            self.path = "/" + "/".join(self.api_url.replace("https://", "").split("/")[1:])
        else:
            self.host = "ark.cn-beijing.volces.com"
            self.path = "/api/v3/chat/completions"

    def chat(self, message: str, system_prompt: str = "You are a helpful assistant.", 
             conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        智能问答核心方法
        :param message: 用户提问
        :param system_prompt: 系统提示词
        :param conversation_history: 对话历史
        :return: 问答结果
        """
        try:
            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加历史对话
            if conversation_history:
                messages.extend(conversation_history)
            
            # 添加当前消息
            messages.append({"role": "user", "content": message})
            
            # 构建请求体
            payload = json.dumps({
                "model": self.model,
                "messages": messages
            })
            
            # 设置请求头
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 发送请求
            conn = http.client.HTTPSConnection(self.host)
            conn.request("POST", self.path, payload, headers)
            res = conn.getresponse()
            data = res.read()
            conn.close()
            
            # 解析响应
            response_data = json.loads(data.decode("utf-8"))
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "content": content,
                    "full_response": response_data
                }
            else:
                return {
                    "success": False,
                    "error": "API响应格式错误",
                    "response": response_data
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"API调用失败: {str(e)}"
            }

    def ask(self, question: str) -> Dict[str, Any]:
        """简化的问答接口"""
        return self.chat(question)

# ======================== 文本分类类 ========================
class TextClassifier:
    """文本分类模型封装类"""
    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.words: Optional[List[str]] = None
        self.word_to_id: Optional[Dict[str, int]] = None
        self.categories = TEXT_CLASSIFY_CATEGORIES
        self.seq_length = TEXT_CLASSIFY_SEQ_LENGTH
        self.load_model()

    def open_file(self, filename: str, mode: str = 'r') -> Any:
        """安全打开文件"""
        return open(filename, mode, encoding='utf-8', errors='ignore')

    def read_vocab(self, vocab_dir: str) -> tuple:
        """读取词汇表"""
        with self.open_file(vocab_dir) as fp:
            words = [i.strip() for i in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id

    def load_model(self) -> None:
        """加载文本分类模型"""
        try:
            # 读取词汇表
            if os.path.exists(TEXT_CLASSIFY_VOCAB_PATH):
                self.words, self.word_to_id = self.read_vocab(TEXT_CLASSIFY_VOCAB_PATH)
                print(f"✅ 词汇表加载成功: {TEXT_CLASSIFY_VOCAB_PATH}")
            else:
                print(f"⚠️  词汇表文件不存在: {TEXT_CLASSIFY_VOCAB_PATH}")
                return
            
            # 加载模型（优先尝试多个可能的路径）
            model_paths = [
                TEXT_CLASSIFY_MODEL_PATH,
                TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_model.h5'),
                TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_validation_best.h5')
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with tf.device('/CPU:0'):
                        self.model = load_model(model_path)
                    print(f"✅ 文本分类模型加载成功: {model_path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print(f"⚠️  文本分类模型文件不存在，相关功能将不可用")
                
        except Exception as e:
            print(f"❌ 加载文本分类模型失败: {e}")
            self.model = None

    def preprocess_text(self, text: str) -> Optional[np.ndarray]:
        """文本预处理：转换为模型输入格式"""
        if not text or not self.word_to_id:
            return None
        
        # 字符转ID
        content = list(text)
        data_id = [self.word_to_id.get(x, 0) for x in content if x in self.word_to_id]
        
        if not data_id:
            return None
        
        # 填充/截断到固定长度
        x_pad = keras.preprocessing.sequence.pad_sequences(
            [data_id], 
            maxlen=self.seq_length, 
            padding='post', 
            truncating='post'
        )
        return x_pad

    def predict(self, text: str) -> Dict[str, Any]:
        """文本分类预测"""
        if self.model is None:
            return {"success": False, "error": "文本分类模型未加载"}
        
        try:
            # 预处理文本
            x_pad = self.preprocess_text(text)
            if x_pad is None:
                return {"success": False, "error": "文本预处理失败"}
            
            # 预测
            with tf.device('/CPU:0'):
                y_pred = self.model.predict(x_pad, verbose=0)
                predicted_class_idx = np.argmax(y_pred[0])
                confidence = float(y_pred[0][predicted_class_idx])
                predicted_class = self.categories[predicted_class_idx]
                
                # 所有类别概率
                probabilities = {
                    self.categories[i]: float(y_pred[0][i]) 
                    for i in range(len(self.categories))
                }
                
                return {
                    "success": True,
                    "category": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities
                }
        except Exception as e:
            return {"success": False, "error": f"预测失败: {str(e)}"}

# ======================== Flask应用初始化 ========================
app = Flask(__name__)
CORS(app)  # 允许跨域

# 初始化核心组件
doubao_api = DoubaoAPI()
text_classifier = TextClassifier()

# ======================== 路由定义 ========================
@app.route('/')
def index():
    """主页（简单的HTML响应）"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>智能问答系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .endpoint { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            code { background: #eee; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>智能问答系统</h1>
        <h2>可用接口</h2>
        <div class="endpoint">
            <strong>智能问答</strong>: POST /api/chat<br>
            请求体: {"message": "你的问题", "system_prompt": "系统提示词", "history": []}
        </div>
        <div class="endpoint">
            <strong>文本分类</strong>: POST /api/classify<br>
            请求体: {"text": "需要分类的文本"}
        </div>
        <div class="endpoint">
            <strong>健康检查</strong>: GET /api/health
        </div>
    </body>
    </html>
    """
    return html_content

@app.route('/api/chat', methods=['POST'])
def chat():
    """智能问答接口"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        system_prompt = data.get('system_prompt', '你是一个智能助手，回答要准确、简洁。')
        conversation_history = data.get('history', None)
        
        if not message:
            return jsonify({"success": False, "error": "消息不能为空"}), 400
        
        result = doubao_api.chat(message, system_prompt, conversation_history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    """文本分类接口"""
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "文本不能为空"}), 400
        
        result = text_classifier.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """系统健康检查接口"""
    return jsonify({
        "status": "healthy",
        "services": {
            "doubao_api": "configured",
            "text_classify": "available" if text_classifier.model else "unavailable",
            "flask": "running"
        },
        "config": {
            "host": FLASK_HOST,
            "port": FLASK_PORT,
            "debug": FLASK_DEBUG
        }
    })

# ======================== 工具函数 ========================
def is_port_available(port: int, host: str = '0.0.0.0') -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False

def find_available_port(start_port: int, max_attempts: int = 10) -> Optional[int]:
    """查找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    return None

# ======================== 启动入口 ========================
if __name__ == '__main__':
    print("="*60)
    print("智能问答系统 - 启动中...")
    print("="*60)
    
    # 检查端口
    port = FLASK_PORT
    if not is_port_available(port):
        print(f"⚠️  端口 {port} 已被占用，正在查找可用端口...")
        available_port = find_available_port(port)
        
        if available_port:
            port = available_port
            print(f"✅ 找到可用端口: {port}")
        else:
            print(f"❌ 未找到可用端口，启动失败")
            sys.exit(1)
    
    # 启动服务
    print(f"\n📡 服务启动成功！")
    print(f"🔗 本地访问: http://localhost:{port}")
    print(f"🌐 外网访问: http://{socket.gethostbyname(socket.gethostname())}:{port}")
    print(f"\n🛑 按 Ctrl+C 停止服务")
    print("="*60)
    
    try:
        app.run(host=FLASK_HOST, port=port, debug=FLASK_DEBUG)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)
