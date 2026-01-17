[__init__.py](https://github.com/user-attachments/files/24687184/__init__.py)[sentiment_analysis.py](https://github.com/user-attachments/files/24687180/sentiment_analysis.py)[translator.py](https://github.com/user-attachments/files/24687176/translator.py)[text_classify.py](https://github.com/user-attachments/files/24687175/text_classify.py)程烁（本人）：
[10_3_1.py](https://github.com/user-attachments/files/24687135/10_3_1.py)[app.py](https://github.com/user-attachments/files/24687123/app.py)
[config.py](https://github.com/user-attachments/files/24687124/config.py)
[test_api.py](https://github.com/user-attachments/files/24687127/test_api.py)
[Up# 文本分类模型加载和推理
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# 配置TensorFlow使用CPU，避免GPU相关错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEXT_CLASSIFY_MODEL_PATH, TEXT_CLASSIFY_VOCAB_PATH, TEXT_CLASSIFY_CATEGORIES, TEXT_CLASSIFY_SEQ_LENGTH

class TextClassifier:
    def __init__(self):
        self.model = None
        self.words = None
        self.word_to_id = None
        self.categories = TEXT_CLASSIFY_CATEGORIES
        self.seq_length = TEXT_CLASSIFY_SEQ_LENGTH
        self.load_model()
    
    def open_file(self, filename, mode='r'):
        """打开文件"""
        return open(filename, mode, encoding='utf-8', errors='ignore')
    
    def read_vocab(self, vocab_dir):
        """读取词汇表"""
        with self.open_file(vocab_dir) as fp:
            words = [i.strip() for i in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id
    
    def load_model(self):
        """加载模型和词汇表"""
        try:
            # 读取词汇表
            if os.path.exists(TEXT_CLASSIFY_VOCAB_PATH):
                self.words, self.word_to_id = self.read_vocab(TEXT_CLASSIFY_VOCAB_PATH)
            else:
                raise FileNotFoundError(f"词汇表文件不存在: {TEXT_CLASSIFY_VOCAB_PATH}")
            
            # 使用CPU加载模型，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 优先加载最佳模型，如果不存在则加载最终模型
                best_model_path = TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_model.h5')
                
                if os.path.exists(best_model_path):
                    self.model = load_model(best_model_path)
                    print(f"文本分类模型加载成功（最佳模型）: {best_model_path}")
                elif os.path.exists(TEXT_CLASSIFY_MODEL_PATH):
                    self.model = load_model(TEXT_CLASSIFY_MODEL_PATH)
                    print(f"文本分类模型加载成功（最终模型）: {TEXT_CLASSIFY_MODEL_PATH}")
                else:
                    # 尝试其他可能的路径
                    alt_path = TEXT_CLASSIFY_MODEL_PATH.replace('my_model.h5', 'best_validation_best.h5')
                    if os.path.exists(alt_path):
                        self.model = load_model(alt_path)
                        print(f"文本分类模型加载成功: {alt_path}")
                    else:
                        raise FileNotFoundError(f"模型文件不存在。尝试过的路径: {best_model_path}, {TEXT_CLASSIFY_MODEL_PATH}")
        except Exception as e:
            print(f"加载文本分类模型失败: {e}")
            self.model = None
    
    def preprocess_text(self, text):
        """预处理文本"""
        if not text:
            return None
        
        # 将文本转换为字符列表
        content = list(text)
        # 转换为ID序列
        data_id = [self.word_to_id.get(x, 0) for x in content if x in self.word_to_id]
        
        if not data_id:
            return None
        
        # 使用 pad_sequences 填充到固定长度（与训练代码保持一致）
        x_pad = keras.preprocessing.sequence.pad_sequences(
            [data_id], 
            maxlen=self.seq_length, 
            padding='post', 
            truncating='post'
        )
        return x_pad
    
    def predict(self, text):
        """预测文本类别"""
        if self.model is None:
            return {"error": "模型未加载"}
        
        try:
            # 预处理文本
            x_pad = self.preprocess_text(text)
            if x_pad is None:
                return {"error": "文本预处理失败"}
            
            # 使用CPU进行预测，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 预测
                y_pred = self.model.predict(x_pad, verbose=0)
                predicted_class_idx = np.argmax(y_pred[0])
                confidence = float(y_pred[0][predicted_class_idx])
                predicted_class = self.categories[predicted_class_idx]
                
                # 返回所有类别的概率
                probabilities = {
                    self.categories[i]: float(y_pred[0][i]) 
                    for i in range(len(self.categories))
                }
                
                return {
                    "category": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities
                }
        except Exception as e:
            error_msg = str(e)
            # 如果是GPU相关错误，提供更友好的提示
            if "stream" in error_msg.lower() or "gpu" in error_msg.lower():
                return {"error": "模型预测时发生设备错误，请稍后重试"}
            return {"error": f"预测失败: {error_msg}"}

# 全局实例
text_classifier = TextClassifier()

loading text_classify.py…]()

[Upload# 10.3.1 文本分类
# 代码10-1 自定义语料预处理函数
import tensorflow as tf
from collections import Counter
from tensorflow import keras
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
# 打开文件
def open_file(filename, mode='r'):
    '''
    filename：表示读取/写入的文件路径
    mode：'r' or 'w'表示读取/写入文件
    '''
    return open(filename, mode, encoding='utf-8', errors='ignore')
# 读取文件数据
def read_file(filename):
    '''
    filename：表示文件路径
    '''
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')  # 按照制表符分割字符串
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels
# 构建词汇表
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    '''
    train_dir：训练集文件的存放路径
    vocab_dir：词汇表的存放路径
    vocab_size：词汇表的大小
    '''
    data_train, lab = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)  # 词袋
    count_pairs = counter.most_common(vocab_size - 1)  # top n
    words, temp = list(zip(*count_pairs))  # 获取key
    words = ['<PAD>'] + list(words)  # 添加一个<PAD>将所有文本pad为同一长度
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
# 读取词汇表
def read_vocab(vocab_dir):
    '''
    vocab_dir：词汇表的存放路径
    '''
    with open_file(vocab_dir) as fp:
        words = [i.strip() for i in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
# 读取分类目录
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # 得到类别与编号相对应的字典，分别为0-9
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id
# 将id表示的内容转换为文字
def to_words(content, words):
    '''
    content：id表示的内容
    words：文本内容
    '''
    return ''.join(words[x] for x in content)
# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    '''
    filename：文件路径
    word_to_id：词汇表
    cat_to_id：类别对应的编号
    max_length：词向量的最大长度
    '''
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # 使用Keras提供的pad_sequences将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转为独热编码（one-hot）表示
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return x_pad, y_pad


# 代码10-2 加载数据并进行预处理

# 设置数据读取、模型、结果保存路径
base_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/data/'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/tmp/'
save_path = os.path.join(save_dir, 'best_validation')

# 若不存在词汇表，则重新建立词汇表
vocab_size = 5000
if not os.path.exists(vocab_dir):
    build_vocab(train_dir, vocab_dir, vocab_size)

# 读取分类目录
categories, cat_to_id = read_category()
# 读取词汇表
words, word_to_id = read_vocab(vocab_dir)
# 词汇表大小
vocab_size = len(words)

# 数据加载
seq_length = 600  # 序列长度

# 获取训练数据
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
# 获取验证数据
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
# 获取测试数据
x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)

# 代码10-3 设置模型参数并构建模型


# 搭建简化的LSTM模型（单层双向LSTM）
def TextRNN():
    model = tf.keras.Sequential()
    # 嵌入层（降低维度以加快训练）
    model.add(tf.keras.layers.Embedding(vocab_size+1, 128, input_length=600, mask_zero=True))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # 单层双向LSTM（简化结构）
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
    ))
    
    # 简化的全连接层
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # 输出层
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# 代码10-4 模型训练（优化版）

# 使用回调函数保存最佳模型
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 创建保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 简化的回调函数
callbacks = [
    # 保存最佳验证准确率的模型
    ModelCheckpoint(
        filepath=os.path.join(save_dir, 'best_model.h5'),
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # 早停机制（减少patience以加快训练）
    EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# 训练参数设置（使用Adam优化器，性能更好）
try:
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = TextRNN()
        # 使用Adam优化器，学习率衰减
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['categorical_accuracy']
        )
except:
    # 如果多GPU策略失败，使用单GPU或CPU
    model = TextRNN()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['categorical_accuracy']
    )

# 模型训练（简化训练轮次）
history = model.fit(
    x_train, y_train, 
    batch_size=128,  # 增大batch size以加快训练
    epochs=10,  # 减少训练轮数
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)
# 设置绘图的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SIMHEI']
# 绘制训练过程
def plot_acc_loss(history):
    '''
    history：模型训练的返回值
    '''
    plt.subplot(121)
    plt.title('准确率趋势图')
    epochs_trained = len(history.history['categorical_accuracy'])
    plt.plot(range(1, epochs_trained+1), history.history['categorical_accuracy'], linestyle='-', color='g', label='训练集')
    plt.plot(range(1, epochs_trained+1), history.history['val_categorical_accuracy'], linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')  # 设置图例
    # x轴按1刻度显示
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.subplot(122)
    plt.title('损失趋势图')
    epochs_trained = len(history.history['loss'])
    plt.plot(range(1, epochs_trained+1), history.history['loss'], linestyle='-', color='g', label='训练集')
    plt.plot(range(1, epochs_trained+1), history.history['val_loss'], linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')
    # x轴按1刻度显示
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.tight_layout()
    plt.show()
    plt.savefig("3.png")
plot_acc_loss(history)

# 代码10-5 查看模型架构并保存模型
model.summary()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存最终模型
final_model_path = os.path.join(save_dir, 'my_model.h5')
model.save(final_model_path)
print(f"最终模型已保存到: {final_model_path}")

# 如果存在最佳模型，也加载它用于测试
best_model_path = os.path.join(save_dir, 'best_model.h5')
if os.path.exists(best_model_path):
    print(f"使用最佳模型进行测试: {best_model_path}")
    model1 = load_model(best_model_path)
else:
    print(f"使用最终模型进行测试: {final_model_path}")
    model1 = model

# 代码10-6 模型测试

# 对测试集进行预测
y_pre = model1.predict(x_test)
# 计算混淆矩阵
confm = confusion_matrix(np.argmax(y_pre, axis=1), np.argmax(y_test, axis=1))
# 打印模型评价
print(classification_report(np.argmax(y_pre, axis=1), np.argmax(y_test, axis=1)))

# 混淆矩阵可视化
plt.figure(figsize=(8, 8), dpi=600)
# 设置绘图的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SIMHEI']
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False, linewidths=.8,
            cmap='YlGnBu')
plt.xlabel('真实标签', size=14)
plt.ylabel('预测标签', size=14)
plt.xticks(np.arange(10)+0.5, categories, size=12)
plt.yticks(np.arange(10)+0.3, categories, size=12)
plt.show()
plt.savefig("1.png")ing 10_3_1.py…]()
李佳音：
[10_3_2.py](https://github.com/user-attachments/files/24687156/10_3_2.py)
[Uploading senti# 情感分析模型加载和推理
import os
import re
import numpy as np
import pandas as pd
import jieba
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import sys

# 配置TensorFlow使用CPU，避免GPU相关错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENTIMENT_MODEL_PATH, SENTIMENT_DICT_PATH, SENTIMENT_SEQ_LENGTH

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.dicts = None
        self.maxlen = SENTIMENT_SEQ_LENGTH
        self.confidence_threshold = 0.5  # 置信度阈值
        self._init_keywords()
        self.load_model()
    
    def _init_keywords(self):
        """初始化情感关键词词典"""
        # 正面情感词（更全面）
        self.positive_words = [
            '好', '棒', '赞', '喜欢', '满意', '不错', '优秀', '完美', '开心', '高兴',
            '爱', '美', '棒极了', '太好了', '推荐', '值得', '满意', '赞', '👍',
            '喜欢', '喜爱', '热爱', '赞美', '称赞', '表扬', '夸奖', '欣赏', '认可',
            '支持', '赞同', '同意', '肯定', '正面', '积极', '乐观', '愉快', '欢乐',
            '兴奋', '激动', '惊喜', '感动', '温暖', '舒适', '安心', '放心', '信任',
            '成功', '胜利', '成就', '进步', '提升', '改善', '优化', '增强', '加强',
            '美好', '精彩', '出色', '卓越', '杰出', '优秀', '优良', '优质', '上乘',
            '超值', '划算', '实惠', '便宜', '经济', '高效', '快速', '便捷', '方便'
        ]
        
        # 负面情感词（更全面）
        self.negative_words = [
            '差', '坏', '烂', '讨厌', '失望', '糟糕', '垃圾', '不好', '伤心', '难过',
            '差劲', '不行', '不推荐', '后悔', '糟糕', '差评', '👎',
            '讨厌', '厌恶', '反感', '嫌弃', '鄙视', '批评', '指责', '抱怨', '埋怨',
            '反对', '拒绝', '否定', '负面', '消极', '悲观', '沮丧', '失落', '绝望',
            '愤怒', '生气', '恼火', '烦躁', '焦虑', '担心', '忧虑', '恐惧', '害怕',
            '失败', '挫折', '困难', '问题', '麻烦', '困扰', '阻碍', '障碍', '缺陷',
            '糟糕', '恶劣', '低劣', '劣质', '次品', '残次', '破损', '损坏', '故障',
            '昂贵', '浪费', '低效', '缓慢', '麻烦', '复杂', '困难', '不便', '不实用'
        ]
        
        # 否定词
        self.negation_words = ['不', '没', '无', '非', '未', '别', '莫', '勿', '否', '没有', '不是', '不能', '不会', '不想', '不要']
        
        # 程度词（增强情感强度）
        self.intensity_words = {
            '非常': 1.5, '特别': 1.5, '极其': 1.8, '十分': 1.4, '相当': 1.3,
            '很': 1.2, '挺': 1.1, '比较': 0.9, '有点': 0.7, '稍微': 0.6,
            '超级': 1.6, '超': 1.5, '太': 1.4, '最': 1.7, '更': 1.2,
            '极其': 1.8, '极度': 1.7, '异常': 1.5, '格外': 1.4
        }
        
        # 停用词（用于文本清洗，注意：不包含否定词）
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '人', '都', '一', '一个',
            '上', '也', '到', '说', '要', '去', '你', '会', '着', '看',
            '自己', '这', '那', '他', '她', '它', '们', '个', '中', '为', '而',
            '与', '及', '或', '但', '如果', '因为', '所以', '虽然', '然而'
        }
    
    def load_model(self):
        """加载模型和词典"""
        try:
            # 使用CPU加载模型，避免GPU相关错误
            with tf.device('/CPU:0'):
                # 加载模型
                if os.path.exists(SENTIMENT_MODEL_PATH):
                    self.model = load_model(SENTIMENT_MODEL_PATH)
                    print(f"情感分析模型加载成功: {SENTIMENT_MODEL_PATH}")
                else:
                    print(f"警告: 情感分析模型文件不存在: {SENTIMENT_MODEL_PATH}")
                    self.model = None
            
            # 加载或创建词典
            if os.path.exists(SENTIMENT_DICT_PATH):
                with open(SENTIMENT_DICT_PATH, 'rb') as f:
                    self.dicts = pickle.load(f)
                print(f"情感分析词典加载成功: {SENTIMENT_DICT_PATH}")
            else:
                print(f"警告: 情感分析词典文件不存在: {SENTIMENT_DICT_PATH}")
                print("将使用简化版情感分析（基于关键词）")
                self.dicts = None
        except Exception as e:
            print(f"加载情感分析模型失败: {e}")
            self.model = None
            self.dicts = None
    
    def clean_text(self, text):
        """清洗文本：去除特殊字符、URL、数字等"""
        if not text:
            return ""
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊符号（保留中文标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：]', '', text)
        # 去除纯数字
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def preprocess_text(self, text):
        """预处理文本"""
        if not text:
            return None
        
        try:
            # 清洗文本
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return None
            
            # 分词
            words = list(jieba.cut(cleaned_text))
            
            # 去除停用词和空字符
            words = [w for w in words if w.strip() and w not in self.stop_words and len(w.strip()) > 0]
            
            if not words:
                return None
            
            if self.dicts is not None:
                # 使用训练时的词典
                word_ids = []
                for word in words:
                    if word in self.dicts.index:
                        word_ids.append(self.dicts.loc[word, 'id'])
                
                if not word_ids:
                    return None
                
                # 填充序列
                sent = sequence.pad_sequences([word_ids], maxlen=self.maxlen)
                return sent
            else:
                # 简化版：基于关键词的情感分析
                return None
        except Exception as e:
            print(f"文本预处理错误: {e}")
            return None
    
    def predict_with_keywords(self, text):
        """基于关键词的简化情感分析（考虑否定词和程度词）"""
        if not text:
            return {"sentiment": "中性", "confidence": 0.5}
        
        # 清洗文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {"sentiment": "中性", "confidence": 0.5}
        
        # 分词
        words = list(jieba.cut(cleaned_text))
        words = [w for w in words if w.strip() and w not in self.stop_words]
        
        if not words:
            return {"sentiment": "中性", "confidence": 0.5}
        
        pos_score = 0.0
        neg_score = 0.0
        
        # 遍历每个词，考虑否定词和程度词的影响
        for i, word in enumerate(words):
            intensity = 1.0  # 默认强度
            negated = False  # 是否被否定
            
            # 检查前面是否有程度词（检查前1-2个词）
            for j in range(max(0, i-2), i):
                if words[j] in self.intensity_words:
                    intensity = self.intensity_words[words[j]]
                    break
            
            # 检查前面是否有否定词（检查前1-3个词，因为否定词可能距离较远）
            for j in range(max(0, i-3), i):
                if words[j] in self.negation_words:
                    negated = True
                    break
            
            # 计算情感分数
            if word in self.positive_words:
                score = 1.0 * intensity
                if negated:
                    neg_score += score  # 否定正面词 = 负面
                else:
                    pos_score += score
            
            elif word in self.negative_words:
                score = 1.0 * intensity
                if negated:
                    pos_score += score  # 否定负面词 = 正面
                else:
                    neg_score += score
        
        # 如果没有找到任何情感词，返回中性
        total_score = pos_score + neg_score
        if total_score == 0:
            return {"sentiment": "中性", "confidence": 0.5}
        
        # 计算置信度（基于分数差异和总分数）
        score_diff = abs(pos_score - neg_score)
        # 如果分数差异明显，置信度更高
        if total_score > 0:
            confidence = 0.5 + min(score_diff / total_score * 0.45, 0.45)
        else:
            confidence = 0.5
        
        # 判断情感（改进判断逻辑，降低阈值以提高准确性）
        # 如果负面分数明显大于正面分数，判定为负面
        if neg_score > pos_score * 1.2:  # 负面分数至少是正面的1.2倍
            return {"sentiment": "负面", "confidence": min(confidence, 0.95)}
        elif pos_score > neg_score * 1.2:  # 正面分数至少是负面的1.2倍
            return {"sentiment": "正面", "confidence": min(confidence, 0.95)}
        elif neg_score > 0 and pos_score == 0:
            # 只有负面词，没有正面词
            return {"sentiment": "负面", "confidence": min(confidence, 0.9)}
        elif pos_score > 0 and neg_score == 0:
            # 只有正面词，没有负面词
            return {"sentiment": "正面", "confidence": min(confidence, 0.9)}
        else:
            # 正面和负面词都存在，根据比例判断
            if neg_score > pos_score:
                return {"sentiment": "负面", "confidence": min(confidence, 0.85)}
            elif pos_score > neg_score:
                return {"sentiment": "正面", "confidence": min(confidence, 0.85)}
            else:
                return {"sentiment": "中性", "confidence": 0.5}
    
    def predict(self, text):
        """预测文本情感"""
        if not text or not text.strip():
            return {"sentiment": "中性", "confidence": 0.5, "method": "default"}
        
        # 如果模型不存在，使用关键词方法
        if self.model is None:
            result = self.predict_with_keywords(text)
            result["method"] = "keywords"
            return result
    
        try:
            # 预处理文本
            x_pad = self.preprocess_text(text)
            if x_pad is None:
                # 如果预处理失败，使用关键词方法
                result = self.predict_with_keywords(text)
                result["method"] = "keywords_fallback"
                return result
        
            # 使用模型预测
            with tf.device('/CPU:0'):
                y_pred = self.model.predict(x_pad, verbose=0)
        
            # 处理模型输出（根据训练代码，模型使用 sigmoid 输出，标签：1=正面，0=负面）
            # 模型输出形状可能是 (1, 1) 或 (1,)
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                # 二分类 softmax 输出（如果模型被修改过）
                negative_prob = float(y_pred[0][0])  # 第一个类别（负面=0）
                positive_prob = float(y_pred[0][1])  # 第二个类别（正面=1）
                
                # 判断情感
                if positive_prob > negative_prob:
                    sentiment = "正面"
                    confidence = positive_prob
                else:
                    sentiment = "负面"
                    confidence = negative_prob
                
                # 获取关键词预测结果用于验证
                keyword_result = self.predict_with_keywords(text)
                
                # 如果模型置信度较低，或者模型预测与关键词预测不一致，需要谨慎处理
                model_uncertain = confidence < self.confidence_threshold or abs(positive_prob - negative_prob) < 0.15
                prediction_conflict = sentiment != keyword_result["sentiment"] and keyword_result["sentiment"] != "中性"
                
                if model_uncertain or prediction_conflict:
                    # 当模型不确定或与关键词预测冲突时，优先参考关键词结果
                    if prediction_conflict and keyword_result["confidence"] > 0.7:
                        # 如果关键词预测置信度高且与模型冲突，优先使用关键词结果
                        sentiment = keyword_result["sentiment"]
                        # 降低模型权重，提高关键词权重
                        combined_confidence = (confidence * 0.3 + keyword_result["confidence"] * 0.7)
                        confidence = combined_confidence
                        return {
                            "sentiment": sentiment,
                            "confidence": float(confidence),
                            "negative_prob": negative_prob,
                            "positive_prob": positive_prob,
                            "method": "model_keywords_combined",
                            "model_sentiment": "正面" if positive_prob > negative_prob else "负面",
                            "keyword_sentiment": keyword_result["sentiment"]
                        }
                    else:
                        # 模型不确定但无冲突，或关键词也不确定，使用加权平均
                        combined_confidence = (confidence * 0.4 + keyword_result["confidence"] * 0.6)
                        if abs(positive_prob - negative_prob) < 0.1:  # 概率接近时，参考关键词结果
                            sentiment = keyword_result["sentiment"]
                        confidence = combined_confidence
                
                return {
                    "sentiment": sentiment,
                    "confidence": float(confidence),
                    "negative_prob": negative_prob,
                    "positive_prob": positive_prob,
                    "method": "model"
                }
            else:
                # 处理 sigmoid 单值输出
                # 输出形状可能是 (1, 1) 或 (1,)
                if len(y_pred.shape) == 2:
                    sentiment_score = float(y_pred[0][0])  # 形状为 (1, 1)
                else:
                    sentiment_score = float(y_pred[0])  # 形状为 (1,)
            
            # 处理 sigmoid 输出（根据训练代码：1=正面，0=负面）
            # sentiment_score 接近 1 表示正面，接近 0 表示负面
            if sentiment_score >= 0.5:
                sentiment = "正面"
                confidence = sentiment_score
            else:
                sentiment = "负面"
                confidence = 1 - sentiment_score
            
            # 获取关键词预测结果用于验证
            keyword_result = self.predict_with_keywords(text)
            
            # 如果模型置信度较低，或者模型预测与关键词预测不一致，需要谨慎处理
            model_uncertain = confidence < self.confidence_threshold or abs(sentiment_score - 0.5) < 0.15
            prediction_conflict = sentiment != keyword_result["sentiment"] and keyword_result["sentiment"] != "中性"
            
            if model_uncertain or prediction_conflict:
                # 当模型不确定或与关键词预测冲突时，优先参考关键词结果
                # 特别是对于明显的负面词（如"伤心"、"难过"），关键词方法更可靠
                if prediction_conflict and keyword_result["confidence"] > 0.7:
                    # 如果关键词预测置信度高且与模型冲突，优先使用关键词结果
                    sentiment = keyword_result["sentiment"]
                    # 降低模型权重，提高关键词权重
                    combined_confidence = (confidence * 0.3 + keyword_result["confidence"] * 0.7)
                    confidence = combined_confidence
                    return {
                        "sentiment": sentiment,
                        "confidence": float(confidence),
                        "score": sentiment_score,
                        "method": "model_keywords_combined",
                        "model_sentiment": "正面" if sentiment_score >= 0.5 else "负面",
                        "keyword_sentiment": keyword_result["sentiment"]
                    }
                else:
                    # 模型不确定但无冲突，或关键词也不确定，使用加权平均
                    combined_confidence = (confidence * 0.4 + keyword_result["confidence"] * 0.6)
                    if abs(sentiment_score - 0.5) < 0.1:  # 接近中性时，参考关键词结果
                        sentiment = keyword_result["sentiment"]
                    confidence = combined_confidence
            
            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "score": sentiment_score,
                "method": "model"
            }
        except Exception as e:
            print(f"模型预测错误: {e}")
            # 出错时回退到关键词方法
            result = self.predict_with_keywords(text)
            result["method"] = "keywords_error_fallback"
            return result

# 全局实例
sentiment_analyzer = SentimentAnalyzer()

ment_analysis.py…]()

盛才厚：
[Uploadin# 机器翻译模型加载和推理
import os
import re
import numpy as np
import tensorflow as tf
import pickle
import sys

# 配置TensorFlow使用CPU（避免GPU相关错误）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRANSLATE_CHECKPOINT_DIR,
    TRANSLATE_DATA_PATH,
    TRANSLATE_TOKENIZER_PATH,
    TRANSLATE_CONFIG_PATH
)


class Translator:
    def __init__(self):
        # 中译英模型（中文→英文）
        self.encoder_zh2en = None
        self.decoder_zh2en = None
        # 英译中模型（英文→中文）
        self.encoder_en2zh = None
        self.decoder_en2zh = None
        
        self.inp_lang = None  # 输入语言（中文）
        self.targ_lang = None  # 目标语言（英文）
        self.max_length_targ = None
        self.max_length_inp = None
        self.units = 1024
        self.embedding_dim = 256
        self.BATCH_SIZE = 1
        self.model_loaded_zh2en = False
        self.model_loaded_en2zh = False
        self.load_model()

    def preprocess_sentence(self, w):
        """预处理句子，与训练时一致"""
        if not w:
            return ""
        w = str(w).strip()
        # 对句子中标点符号前后加空格
        w = re.sub(r'([?.!,])', r' \1 ', w)
        # 将句子中多空格去重
        w = re.sub(r"[' ']+", ' ', w)
        # 给句子加上开始和结束标记
        w = '<start> ' + w.strip() + ' <end>'
        return w

    def load_model(self):
        """加载模型（结构与训练时完全一致）"""
        try:
            print(f"[DEBUG] Tokenizer路径: {TRANSLATE_TOKENIZER_PATH}")
            print(f"[DEBUG] Checkpoint路径: {TRANSLATE_CHECKPOINT_DIR}")

            # 1. 加载Tokenizer
            if not os.path.exists(TRANSLATE_TOKENIZER_PATH):
                raise FileNotFoundError(f"Tokenizer文件不存在: {TRANSLATE_TOKENIZER_PATH}")

            with open(TRANSLATE_TOKENIZER_PATH, 'rb') as f:
                tokenizer_data = pickle.load(f)
                self.inp_lang = tokenizer_data['inp_lang']
                self.targ_lang = tokenizer_data['targ_lang']
                self.max_length_targ = tokenizer_data['max_length_targ']
                self.max_length_inp = tokenizer_data['max_length_inp']
                self.embedding_dim = tokenizer_data.get('embedding_dim', 256)
                self.units = tokenizer_data.get('units', 1024)

            # 2. 检查Checkpoint目录
            if not os.path.exists(TRANSLATE_CHECKPOINT_DIR):
                raise FileNotFoundError(f"Checkpoint目录不存在: {TRANSLATE_CHECKPOINT_DIR}")

            # ===== 模型结构定义（与训练一致） =====
            class Encoder(tf.keras.Model):
                def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
                    super().__init__()
                    self.batch_sz = batch_sz
                    self.enc_units = enc_units
                    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
                    self.bigru = tf.keras.layers.Bidirectional(
                        tf.keras.layers.GRU(
                            enc_units // 2,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer='glorot_uniform',
                            dropout=0.2,
                            recurrent_dropout=0.2
                        )
                    )
                    self.state_proj = tf.keras.layers.Dense(enc_units, activation='tanh')

                def call(self, x, hidden, training=False):
                    x = self.embedding(x)
                    output, f_state, b_state = self.bigru(x, initial_state=[hidden, hidden], training=training)
                    state = self.state_proj(tf.concat([f_state, b_state], axis=-1))
                    return output, state

                def initialize_hidden_state(self):
                    return tf.zeros((self.batch_sz, self.enc_units // 2))

            class BahdanauAttention(tf.keras.layers.Layer):
                def __init__(self, units):
                    super().__init__()
                    self.W1 = tf.keras.layers.Dense(units)
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V = tf.keras.layers.Dense(1)

                def call(self, query, values):
                    hidden_with_time_axis = tf.expand_dims(query, 1)
                    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
                    attention_weights = tf.nn.softmax(score, axis=1)
                    context_vector = attention_weights * values
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, attention_weights

            class Decoder(tf.keras.Model):
                def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
                    super().__init__()
                    self.batch_sz = batch_sz
                    self.dec_units = dec_units
                    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
                    self.gru = tf.keras.layers.GRU(
                        dec_units,
                        return_sequences=True,
                        return_state=True,
                        recurrent_initializer='glorot_uniform',
                        dropout=0.2,
                        recurrent_dropout=0.2
                    )
                    self.fc_mid = tf.keras.layers.Dense(dec_units, activation='relu')
                    self.fc = tf.keras.layers.Dense(vocab_size)
                    self.attention = BahdanauAttention(dec_units)

                def call(self, x, hidden, enc_output, training=False):
                    context_vector, attention_weights = self.attention(hidden, enc_output)
                    x = self.embedding(x)
                    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                    output, state = self.gru(x, initial_state=hidden, training=training)
                    output = tf.reshape(output, (-1, output.shape[2]))
                    output = self.fc_mid(output)
                    x = self.fc(output)
                    return x, state, attention_weights

            # ===== 创建中译英模型实例（中文→英文） =====
            vocab_inp_size = len(self.inp_lang.word_index) + 1
            vocab_tar_size = len(self.targ_lang.word_index) + 1

            self.encoder_zh2en = Encoder(vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
            self.decoder_zh2en = Decoder(vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)

            # ===== 创建英译中模型实例（英文→中文，交换vocab） =====
            # 注意：英译中需要交换输入输出vocab
            self.encoder_en2zh = Encoder(vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)
            self.decoder_en2zh = Decoder(vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)

            # ===== 加载Checkpoint（中译英模型） =====
            latest_checkpoint = tf.train.latest_checkpoint(TRANSLATE_CHECKPOINT_DIR)
            if not latest_checkpoint:
                raise FileNotFoundError(f"未找到checkpoint文件: {TRANSLATE_CHECKPOINT_DIR}")

            # 加载中译英模型
            checkpoint_zh2en = tf.train.Checkpoint(encoder=self.encoder_zh2en, decoder=self.decoder_zh2en)
            status_zh2en = checkpoint_zh2en.restore(latest_checkpoint)
            status_zh2en.expect_partial()
            self.model_loaded_zh2en = True
            print(f"[INFO] 中译英模型加载成功: {latest_checkpoint}")
            
            # 尝试加载英译中模型
            # 注意：由于vocab size不同，无法直接使用同一个checkpoint
            # 我们尝试创建一个反向模型，但权重需要重新训练或手动映射
            # 这里我们尝试加载，如果失败则使用随机初始化的权重（效果较差，但可以运行）
            try:
                checkpoint_en2zh = tf.train.Checkpoint(
                    encoder=self.encoder_en2zh, 
                    decoder=self.decoder_en2zh
                )
                # 尝试从同一个checkpoint加载（会失败，因为vocab size不匹配）
                # 但expect_partial会忽略不匹配的部分，模型会使用随机初始化的权重
                status_en2zh = checkpoint_en2zh.restore(latest_checkpoint)
                status_en2zh.expect_partial()
                # 检查是否有任何权重被加载
                # 由于vocab size不同，embedding和fc层无法加载，但GRU等层可能可以共享
                # 这里我们标记为已加载，但实际效果可能不理想
                self.model_loaded_en2zh = True
                print(f"[INFO] 英译中模型已创建（部分权重可能未加载，效果可能不理想）")
                print(f"[INFO] 建议：如需高质量英译中，请训练反向模型")
            except Exception as e:
                print(f"[WARNING] 英译中模型创建失败: {e}")
                print(f"[INFO] 英译中将使用简化词典翻译")
                self.model_loaded_en2zh = False
            
            print(f"[INFO] 模型支持方向: 中文 → 英文 (zh2en): {'✓' if self.model_loaded_zh2en else '✗'}")
            print(f"[INFO] 模型支持方向: 英文 → 中文 (en2zh): {'✓' if self.model_loaded_en2zh else '✗'}")

        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.encoder_zh2en = None
            self.decoder_zh2en = None
            self.encoder_en2zh = None
            self.decoder_en2zh = None
            self.model_loaded_zh2en = False
            self.model_loaded_en2zh = False

    def evaluate(self, sentence, direction='zh2en'):
        """模型推理
        Args:
            sentence: 待翻译的句子
            direction: 翻译方向，'zh2en' 表示中文→英文，'en2zh' 表示英文→中文
        """
        try:
            # 根据方向选择模型和语言
            if direction == 'zh2en':
                # 中译英：使用 inp_lang（中文）作为输入，targ_lang（英文）作为输出
                if not self.model_loaded_zh2en or self.encoder_zh2en is None or self.decoder_zh2en is None:
                    return None
                
                encoder = self.encoder_zh2en
                decoder = self.decoder_zh2en
                input_lang = self.inp_lang
                output_lang = self.targ_lang
                max_input_len = self.max_length_inp
                max_output_len = self.max_length_targ
                
            elif direction == 'en2zh':
                # 英译中：使用 targ_lang（英文）作为输入，inp_lang（中文）作为输出
                if not self.model_loaded_en2zh or self.encoder_en2zh is None or self.decoder_en2zh is None:
                    return None
                
                encoder = self.encoder_en2zh
                decoder = self.decoder_en2zh
                input_lang = self.targ_lang  # 英文作为输入
                output_lang = self.inp_lang  # 中文作为输出
                max_input_len = self.max_length_targ  # 英文的最大长度
                max_output_len = self.max_length_inp  # 中文的最大长度
            else:
                return None

            sentence = self.preprocess_sentence(sentence)
            inputs = [input_lang.word_index.get(i, 0) for i in sentence.split() if i]
            if not inputs:
                return ""
            
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_input_len, padding='post')
            inputs = tf.convert_to_tensor(inputs)

            hidden = encoder.initialize_hidden_state()
            enc_out, enc_hidden = encoder(inputs, hidden, training=False)
            dec_hidden = enc_hidden

            start_token = output_lang.word_index['<start>'] if '<start>' in output_lang.word_index else 1
            dec_input = tf.expand_dims([start_token], 0)

            result = ""
            for _ in range(max_output_len):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out, training=False)
                predicted_id = tf.argmax(predictions[0]).numpy()
                predicted_word = output_lang.index_word.get(predicted_id, "")
                if predicted_word == '<end>':
                    break
                if predicted_word != '<start>':  # 跳过开始标记
                    result += predicted_word + " "
                dec_input = tf.expand_dims([predicted_id], 0)

            return result.strip()

        except Exception as e:
            print(f"[ERROR] 推理失败 ({direction}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def simple_translate(self, text, direction='zh2en'):
        """固定词典翻译（降级用）
        Args:
            text: 待翻译的文本
            direction: 翻译方向，'zh2en' 或 'en2zh'
        """
        if not text:
            return ""
        
        if direction == 'zh2en':
            # 中译英词典（按长度降序，优先匹配长短语）
            common_dict = {
                '很高兴见到你': 'Nice to meet you',
                '早上好': 'Good morning', 
                '晚上好': 'Good evening',
                '不客气': "You're welcome",
                '我爱你': 'I love you',
                '你好': 'Hello', 
                '谢谢': 'Thank you', 
                '再见': 'Goodbye',
                '是的': 'Yes', 
                '不是': 'No', 
                '对不起': 'Sorry',
                '请': 'Please',
                '谢谢': 'Thanks',
                '好的': 'OK',
                '没问题': 'No problem',
                '当然': 'Of course'
            }
            result = text
            # 按长度降序排序，优先匹配长短语
            for zh, en in sorted(common_dict.items(), key=lambda x: len(x[0]), reverse=True):
                result = result.replace(zh, en)
            return result
        else:
            # 英译中词典（按长度降序，优先匹配长短语）
            common_dict = {
                'Nice to meet you': '很高兴见到你',
                'Good morning': '早上好', 
                'Good evening': '晚上好',
                "You're welcome": '不客气',
                'I love you': '我爱你',
                'Thank you': '谢谢', 
                'Goodbye': '再见',
                'Hello': '你好',
                'Yes': '是的', 
                'No': '不是', 
                'Sorry': '对不起',
                'Please': '请',
                'Thanks': '谢谢',
                'OK': '好的',
                'No problem': '没问题',
                'Of course': '当然'
            }
            result = text
            # 按长度降序排序，优先匹配长短语
            for en, zh in sorted(common_dict.items(), key=lambda x: len(x[0]), reverse=True):
                result = result.replace(en, zh)
            return result

    def translate(self, text, direction='zh2en'):
        """对外翻译接口
        Args:
            text: 待翻译的文本
            direction: 翻译方向，'zh2en' 表示中文→英文，'en2zh' 表示英文→中文
        Returns:
            dict: 包含 original, translated, method, direction 的字典
        """
        if direction not in ['zh2en', 'en2zh']:
            return {
                "original": text,
                "translated": "不支持的方向，请使用 'zh2en' 或 'en2zh'",
                "method": "错误",
                "direction": direction
            }

        # 中译英：使用模型翻译
        if direction == 'zh2en':
            if not self.model_loaded_zh2en:
                # 模型未加载，使用简化翻译
                return {
                    "original": text,
                    "translated": self.simple_translate(text, direction),
                    "method": "模型未加载，使用简化词典翻译",
                    "direction": direction
                }

            model_result = self.evaluate(text, direction)
            if model_result:
                return {
                    "original": text,
                    "translated": model_result,
                    "method": "Seq2Seq模型翻译",
                    "direction": direction
                }
            else:
                # 模型推理失败，使用简化翻译
                return {
                    "original": text,
                    "translated": self.simple_translate(text, direction),
                    "method": "模型推理失败，使用简化词典翻译",
                    "direction": direction
                }
        
        # 英译中：尝试使用模型翻译，如果失败则使用简化翻译
        else:  # direction == 'en2zh'
            # 注意：由于vocab size不同，英译中模型可能无法从checkpoint正确加载
            # 如果模型已加载，尝试使用；否则直接使用简化翻译
            if self.model_loaded_en2zh:
                model_result = self.evaluate(text, direction)
                if model_result and model_result.strip():
                    # 检查结果是否合理（不是空字符串或只有标点）
                    return {
                        "original": text,
                        "translated": model_result,
                        "method": "Seq2Seq模型翻译（部分权重可能未加载）",
                        "direction": direction
                    }
            
            # 模型未加载或推理失败，使用简化翻译
            return {
                "original": text,
                "translated": self.simple_translate(text, direction),
                "method": "简化词典翻译" + ("（模型权重未正确加载）" if self.model_loaded_en2zh else "（模型未加载）"),
                "direction": direction,
                "note": "如需高质量英译中，请训练反向模型或使用专门的英译中checkpoint"
            }


# 全局实例
translator = Translator()g translator.py…]()

# 10.4 任务：基于Seq2Seq的机器翻译
# 代码10-12 语料预处理
import re
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
from tqdm import  tqdm
import numpy as np
# 准备数据集
def preprocess_sentence(w):   
    '''
    w：句子
    '''
    w = re.sub(r'([?.!,])', r' \1 ', w)  # 对句子中标点符号前后加空格
    w = re.sub(r"[' ']+", ' ', w)  # 将句子中多空格去重
    w = '<start> ' + w + ' <end>'  # 给句子加上开始和结束标记，以便模型预测
    return w

en_sentence = 'I like this book'
sp_sentence = '我喜欢这本书'
print('预处理前的输出为：', '\n', preprocess_sentence(en_sentence))
print('预处理前的输出为：', '\n', str(preprocess_sentence(sp_sentence)), 'utf-8', '\n')

# 清理句子，删除重音符号，返回格式为[英文，中文]的单词对
def create_dataset(path, num_examples):
    '''
    path：文件路径
    num_examples：选用的数据量
    '''
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)

path_to_file = '/root/autodl-tmp/NLP/nlp_deeplearn/data/en-ch.txt'  # 读取文件的路径
en, sp = create_dataset(path_to_file, None)  # 整合并读取数据

# 句子的最大长度
def max_length(tensor):
    '''
    tensor：文本构成的张量
    '''
    return max(len(t) for t in tensor)

# tokenize函数是对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示
def tokenize(lang):
    '''
    lang：待处理的文本
    '''
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

# 创建清理的输入输出对
def load_dataset(path, num_examples=None):
    '''
    path：文件路径
    num_examples：选用的数据量
    '''
    # 建立索引，并输入已经清洗过的词语，输出词语对
    targ_lang, inp_lang = create_dataset(path, num_examples) 
    # 建立中文句子的词向量，对所有张量进行填充，使句子的维度一样
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)   
    # 建立英文句子的词向量，对所有张量进行填充，使句子的维度一样
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)  
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

num_examples = 2000  # 词表的大小（词量）
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, 
                                                                num_examples)
# 计算目标张量的最大长度（max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(
    input_tensor) 

# 采用8: 2的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2) 

# 验证数据正确性，也就是输出词与词语映射索引的表示
def convert(lang, tensor):
    '''
    lang：待处理的文本
    tensor：文本构成的张量
    '''
    for t in tensor:
        if t != 0:    
            print ('%d ----> %s' % (t, lang.index_word[t]))

print('预处理前的输出为：')
print('输入语言：词映射索引')
convert(inp_lang, input_tensor_train[0])
print('目标语言：词语映射索引')
convert(targ_lang, target_tensor_train[0])

# 创建tf.data数据集
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64          # 减小 batch，有利于收敛
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256      # 提高词向量维度，增强表达能力
units = 512              # 保持 512，避免太慢
vocab_inp_size = len(inp_lang.word_index)+1  # 输入词表的大小
vocab_tar_size = len(targ_lang.word_index)+1  # 输出词表的大小
dataset = tf.data.Dataset.from_tensor_slices((
    input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # 构建训练集
example_input_batch, example_target_batch = next(iter(dataset))



# 代码10-13 构建机器翻译模型
# 双向编码器（Bi-GRU）
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # 输入嵌入
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        # 双向 GRU
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform',
                dropout=0.2,
                recurrent_dropout=0.2
            )
        )
        # 把前向/后向状态拼接后降维回 enc_units
        self.state_proj = tf.keras.layers.Dense(self.enc_units, activation='tanh')

    def call(self, x, hidden):
        x = self.embedding(x)
        # bigru 返回：output, forward_state, backward_state
        output, f_state, b_state = self.bigru(x, initial_state=[hidden, hidden])
        # 拼接两个方向的 hidden，再投影回 enc_units
        h_cat = tf.concat([f_state, b_state], axis=-1)
        state = self.state_proj(h_cat)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# 构建编码器网络结构    
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)  
print('编码器输出形状：', '\n', ' (batch size, sequence length, units) {}'.format(sample_output.shape))
print('编码器隐藏状态形状：', '\n', ' (batch size, units) {}'.format(sample_hidden.shape))

# 注意力机制（保持原来的 BahdanauAttention 定义即可）
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# 解码器：单层 GRU + 中间全连接
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            dropout=0.2,
            recurrent_dropout=0.2
        )
        # 新增一个中间全连接层
        self.fc_mid = tf.keras.layers.Dense(self.dec_units, activation='relu')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        # 先通过中间层
        output = self.fc_mid(output)
        x = self.fc(output)
        return x, state, attention_weights

# 构建解码器网络结构
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)  
sample_decoder_output, states, attention_weight = decoder(
    tf.random.uniform((BATCH_SIZE, 1), maxval=vocab_tar_size, dtype=tf.int32),
    sample_hidden,
    sample_output
)
print('解码器输出形状：', '\n', ' (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


# 代码10-14 定义优化器及损失函数（优化版）
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 带 label smoothing 的损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)



# 代码10-15 训练模型

# 检查点（基于对象的保存），准备保存训练模型
checkpoint_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/tmp/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)  # 保存模型
# 训练模型
def train(inp, targ, enc_hidden):
    '''
    inp：批次
    targ：标签
    enc_hidden：隐藏样本
    '''
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)  # 构建编码器
        dec_hidden = enc_hidden  
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出传送至解码器
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)  # 使用教师强制
        loss = loss / int(targ.shape[1])  # 计算平均损失
    batch_loss = loss.numpy()  # 将损失转换为numpy数组
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# 开始训练（适度增加轮次以提升准确率）
EPOCHS = 30  # 适当增加训练轮数，通常能明显提升翻译质量
loss = []

for epoch in tqdm(range(EPOCHS)):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()  # 初始化隐藏层
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train(inp, targ, enc_hidden)
        total_loss += batch_loss
        if batch % 50 == 0:  # 减少打印频率
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))
            loss.append(round(batch_loss, 3))
    
    print('Epoch {} 平均损失: {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    
    # 每5轮保存一次模型
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        print('保存模型检查点')

# 损失趋势可视化

plt.rcParams['font.sans-serif'] = ['SIMHEI']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 对字符进行显示设置
if loss:  # 只有当有损失数据时才绘图
    plt.plot(list(range(1, len(loss)+1)), loss)  # 将损失值绘制成折线图
    plt.title('损失趋势图', fontsize=16)  # 设置折线图标题为损失趋势图
    plt.xlabel('迭代次数')  # 将x轴标签设置为迭代次数
    plt.ylabel('损失值')  # 将y轴标签设置为损失值
    plt.show()  # 将图形进行展示
    plt.savefig("10_4.png")


# 代码10-16 使用模型进行语句翻译

# 优化的翻译函数（支持beam search）
def evaluate(sentence, beam_width=1):
    '''
    sentence：需要翻译的句子
    beam_width：beam search的宽度（1表示贪心搜索）
    '''
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index.get(i, 0) for i in sentence.split(' ') if i in inp_lang.word_index]
    if not inputs:
        return '', sentence, attention_plot
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = tf.zeros((1, units))
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    if beam_width == 1:
        # 贪心搜索
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            if predicted_id in targ_lang.index_word:
                predicted_word = targ_lang.index_word[predicted_id]
                if predicted_word == '<end>':
                    break
                result += predicted_word + ' '
            else:
                break
            dec_input = tf.expand_dims([predicted_id], 0)
    else:
        # 简化的beam search（可以进一步优化）
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            # 获取top-k预测
            top_k = tf.nn.top_k(predictions[0], k=min(beam_width, len(targ_lang.word_index)))
            predicted_id = top_k.indices[0].numpy()
            if predicted_id in targ_lang.index_word:
                predicted_word = targ_lang.index_word[predicted_id]
                if predicted_word == '<end>':
                    break
                result += predicted_word + ' '
            else:
                break
            dec_input = tf.expand_dims([predicted_id], 0)
    
    return result, sentence, attention_plot

# 执行翻译▲
def translate(sentence):
    '''
    sentence：要翻译的句子
    '''
    result, sentence, attention_plot = evaluate(sentence)
    print('输入：%s' % (sentence))
    print('翻译结果：{}'.format(result))

print(translate('我生病了。'))
print(translate('为什么不？'))
print(translate('让我一个人呆会儿。'))
print(translate('打电话回家！'))
print(translate('我了解你。'))

# ===== 新增：保存训练结果以便在qa_system中使用 =====
import pickle

# 保存tokenizer和模型参数
translate_save_dir = '/root/autodl-tmp/NLP/nlp_deeplearn/tmp/'
os.makedirs(translate_save_dir, exist_ok=True)

# 保存tokenizer
tokenizer_save_path = os.path.join(translate_save_dir, 'translate_tokenizers.pkl')
with open(tokenizer_save_path, 'wb') as f:
    pickle.dump({
        'inp_lang': inp_lang,
        'targ_lang': targ_lang,
        'max_length_targ': max_length_targ,
        'max_length_inp': max_length_inp,
        'vocab_inp_size': vocab_inp_size,
        'vocab_tar_size': vocab_tar_size,
        'embedding_dim': embedding_dim,
        'units': units
    }, f)
print(f"\n翻译模型tokenizer已保存到: {tokenizer_save_path}")

# 保存模型配置信息
config_save_path = os.path.join(translate_save_dir, 'translate_config.txt')
with open(config_save_path, 'w', encoding='utf-8') as f:
    f.write(f"max_length_targ={max_length_targ}\n")
    f.write(f"max_length_inp={max_length_inp}\n")
    f.write(f"vocab_inp_size={vocab_inp_size}\n")
    f.write(f"vocab_tar_size={vocab_tar_size}\n")
    f.write(f"embedding_dim={embedding_dim}\n")
    f.write(f"units={units}\n")
    f.write(f"checkpoint_dir={checkpoint_dir}\n")
print(f"翻译模型配置已保存到: {config_save_path}")

print("\n训练结果已保存，现在可以在qa_system中加载使用了。")
print(f"检查点目录: {checkpoint_dir}")
print(f"请确保在qa_system中使用最新的checkpoint进行加载。")

殷小曼：
# API模块初始化文件


# 豆包 API 集成
import http.client
import json
from config import DOUBAO_API_URL, DOUBAO_API_KEY, DOUBAO_MODEL

class DoubaoAPI:
    def __init__(self):
        self.api_url = DOUBAO_API_URL
        self.api_key = DOUBAO_API_KEY
        self.model = DOUBAO_MODEL
        # 从URL中提取主机
        if "https://" in self.api_url:
            self.host = self.api_url.replace("https://", "").split("/")[0]
            self.path = "/" + "/".join(self.api_url.replace("https://", "").split("/")[1:])
        else:
            self.host = "ark.cn-beijing.volces.com"
            self.path = "/api/v3/chat/completions"
    
    def chat(self, message, system_prompt="You are a helpful assistant.", conversation_history=None):
        """
        调用豆包API进行对话
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            conversation_history: 对话历史记录
        """
        try:
            # 构建消息列表
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # 添加历史对话
            if conversation_history:
                messages.extend(conversation_history)
            
            # 添加当前消息
            messages.append({
                "role": "user",
                "content": message
            })
            
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
    
    def ask(self, question):
        """简单问答接口"""
        return self.chat(question)

# 全局实例
doubao_api = DoubaoAPI()

创意功能：
[__init__.py](https://github.com/user-attachments/files/24687188/__init__.py)
[creative_features.py](https://github.com/user-attachments/files/24687189/creative_features.py)

[index.html](https://github.com/user-attachments/files/24687191/index.html)
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能问答系统</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <!-- 背景装饰元素 -->
    <div class="anime-decorations">
        <div class="floating-cat cat-1">🐱</div>
        <div class="floating-cat cat-2">✨</div>
        <div class="floating-cat cat-3">⭐</div>
        <div class="floating-cat cat-4">💫</div>
        <div class="floating-cat cat-5">🌟</div>
        <div class="floating-cat cat-6">💖</div>
        <div class="floating-cat cat-7">🐾</div>
        <div class="floating-cat cat-8">🌸</div>
    </div>
    
    <div class="container">
        <header>
            <div class="header-content">
                <img src="{{ url_for('static', filename='images/cat-icon.svg') }}" alt="智能助手" class="header-icon" onerror="this.style.display='none'; this.nextElementSibling.style.display='inline-block';">
                <span class="header-icon-fallback" style="display:none;">🐱</span>
                <div class="header-text">
                    <h1>多功能智能问答系统</h1>
                    <p class="subtitle">集成豆包API、文本分类、情感分析、机器翻译等功能</p>
                </div>
            </div>
        </header>

        <div class="main-wrapper">
            <!-- 左侧功能选择标签 -->
            <div class="sidebar">
                <div class="tabs">
                    <button class="tab-btn active" data-tab="chat">💬 智能问答</button>
                    <button class="tab-btn" data-tab="classify">📝 文本分类</button>
                    <button class="tab-btn" data-tab="sentiment">😊 情感分析</button>
                    <button class="tab-btn" data-tab="translate">🌐 机器翻译</button>
                    <button class="tab-btn" data-tab="creative">✨ 创意功能</button>
                </div>
            </div>

            <!-- 右侧主内容区域 -->
            <div class="main-content">
                <!-- 智能问答面板 -->
            <div class="tab-content active" id="chat">
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message system cat-message">
                            <div class="cat-ears">
                                <span class="ear-left">🐱</span>
                                <span class="ear-right">🐱</span>
                            </div>
                            <p>👋 欢迎使用哈基米智能问答系统！我是您的AI助手，可以回答各种问题。请随时向我提问！</p>
                            <div class="cat-tail">🐾</div>
                        </div>
                    </div>
                    <div class="chat-input-area">
                        <div class="input-wrapper">
                            <span class="cat-emoji-input">🐱</span>
                            <textarea id="chatInput" placeholder="输入您的问题..."></textarea>
                        </div>
                        <button id="sendBtn" class="btn-send">发送 🐾</button>
                    </div>
                </div>
            </div>

            <!-- 文本分类面板 -->
            <div class="tab-content" id="classify">
                <div class="feature-panel">
                    <h3>文本分类</h3>
                    <p>将文本分类到以下类别：体育、财经、房产、家居、教育、科技、时尚、时政、游戏、娱乐</p>
                    <textarea id="classifyInput" placeholder="输入要分类的文本..."></textarea>
                    <button class="btn-primary" onclick="classifyText()">分类</button>
                    <div id="classifyResult" class="result-box"></div>
                </div>
            </div>

            <!-- 情感分析面板 -->
            <div class="tab-content" id="sentiment">
                <div class="feature-panel">
                    <h3>情感分析</h3>
                    <p>分析文本的情感倾向（正面/负面）</p>
                    <textarea id="sentimentInput" placeholder="输入要分析的文本..."></textarea>
                    <button class="btn-primary" onclick="analyzeSentiment()">分析</button>
                    <div id="sentimentResult" class="result-box"></div>
                </div>
            </div>

            <!-- 机器翻译面板 -->
            <div class="tab-content" id="translate">
                <div class="feature-panel">
                    <h3>机器翻译</h3>
                    <p>支持中英文互译</p>
                    <div class="translate-controls">
                        <label>
                            <input type="radio" name="direction" value="zh2en" checked> 中文 → 英文
                        </label>
                        <label>
                            <input type="radio" name="direction" value="en2zh"> 英文 → 中文
                        </label>
                    </div>
                    <textarea id="translateInput" placeholder="输入要翻译的文本..."></textarea>
                    <button class="btn-primary" onclick="translateText()">翻译</button>
                    <div id="translateResult" class="result-box"></div>
                </div>
            </div>

            <!-- 创意功能面板 -->
            <div class="tab-content" id="creative">
                <div class="feature-panel">
                    <h3>创意功能</h3>
                    <div class="creative-buttons">
                        <button class="btn-secondary" onclick="extractKeywords()">🔑 提取关键词</button>
                        <button class="btn-secondary" onclick="generateSummary()">📄 文本摘要</button>
                        <button class="btn-secondary" onclick="wordFrequency()">📊 词频统计</button>
                        <button class="btn-secondary" onclick="textStatistics()">📈 文本统计</button>
                        <button class="btn-secondary" onclick="detectLanguage()">🌍 语言检测</button>
                    </div>
                    <textarea id="creativeInput" placeholder="输入文本以使用创意功能..."></textarea>
                    <div id="creativeResult" class="result-box"></div>
                </div>
            </div>
            </div>
        </div>

        <footer>
            <p>系统状态: <span id="systemStatus">检查中...</span></p>
        </footer>
    </div>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>


