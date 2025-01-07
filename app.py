import streamlit as st
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import re
import torch.nn.functional as F
import pandas as pd

#运行指令：streamlit run app.py

# 定义 LSTM 模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后时间步的输出
        x = self.fc(self.dropout(lstm_out))
        return x

# 加载模型
vocab_size = 40157  # 需要与训练时的一致
embedding_dim = 100
hidden_dim = 128
output_dim = 2
dropout_rate = 0.3

model = LSTMClassifier(vocab_size + 1, embedding_dim, hidden_dim, output_dim, dropout_rate)
model.load_state_dict(torch.load('./lstm_model.pth'))
model.eval()

# 加载词汇表
checkpoint = torch.load('./processed_data.pt')  # 假设你在这里保存了词汇表
word_to_idx = checkpoint['word_to_idx']

# 清洗和tokenization函数
def remove_punc(text):
    return ''.join([char for char in text if char not in punctuation])

def remove_stop(text):
    stops = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word.lower() not in stops])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = remove_punc(text)
    text = remove_stop(text)
    tokens = word_tokenize(text)
    return tokens

# 填充序列
max_len = 200
def truncate_and_pad(sequences, max_len):
    if len(sequences) > max_len:  # 裁剪
        return sequences[:max_len]
    else:  # 填充
        return sequences + [0] * (max_len - len(sequences))

# Streamlit 界面设置
st.title("Text Classification with LSTM")
st.write("Enter a text below for classification.")

# 用户输入
user_input = st.text_area("Input Text", "This is an example sentence for inference.")

# 按钮触发推理
if st.button('Classify Text'):
    if user_input:
        # 数据预处理
        tokens = preprocess(user_input)
        numerical = [word_to_idx.get(word, 0) for word in tokens]  # 使用 get 防止找不到词时返回0
        numerical = truncate_and_pad(numerical, max_len)
        input_tensor = torch.tensor([numerical], dtype=torch.long)

        # 进行推理
        with torch.no_grad():
            output = model(input_tensor)

            # 计算 softmax 以获取每个类别的概率
            probabilities = F.softmax(output, dim=1)

            # 获取预测的类别（最大概率）及对应的置信度
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0, predicted_class]  # 对应预测类别的置信度

        # 输出推理结果
        if predicted_class.item() == 0:
            st.subheader(f"Result: Human-written")
        else:
            st.subheader(f"Result: AI-written")

        # 输出置信度
        st.write(f"Confidence: {confidence.item():.4f}")
    else:
        st.error("Please enter some text to classify.")

