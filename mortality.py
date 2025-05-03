import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('XGBoost.pkl')  # 加载训练好的XGBoost模型

# Define the feature options
Initial_cardiac_rhythm_options = {
    0: 'Non-shockable',  # 非可除颤心律
    1: 'Shockable',  # 可除颤心律
}

Bystander_CPR_options = {
    0: 'No',  # 没有旁观者CPR
    1: 'Yes',  #有旁观者CPR
}


# Streamlit UI
st.title(" 28-day mortality predictor")  # 预测器

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# CPR_time input
CPR_time = st.sidebar.number_input("CPR_time:", min_value=2, max_value=56, value=50)  # CPR_time输入框

# Initial_cardiac_rhythm input
Initial_cardiac_rhythm = st.sidebar.selectbox("Initial_cardiac_rhythm:", options=list(Initial_cardiac_rhythm_options.keys()), format_func=lambda x: Initial_cardiac_rhythm_options[x])  # 类型选择框

# Bystander_CPR input
Bystander_CPR = st.sidebar.selectbox("Bystander_CPR:", options=list(Bystander_CPR_options.keys()), format_func=lambda x: Bystander_CPR_options[x])  # 类型选择框

# SOFA_score input
SOFA_score = st.sidebar.number_input("SOFA_score:", min_value=0, max_value=14, value=4)  # 输入框

# sTREM_1 input
sTREM_1 = st.sidebar.number_input("sTREM_1:", min_value=33.12, max_value=594.09, value=300.00)  # 输入框

# NSE input
NSE = st.sidebar.number_input("NSE:", min_value=1.30, max_value=68.52, value=10.00)  # 输入框

# IL_6 input
IL_6 = st.sidebar.number_input("IL_6:", min_value=20.27, max_value=181.45, value=30.00)  # 输入框

# IL_10 input
IL_10 = st.sidebar.number_input("IL_10:", min_value=1.06, max_value=38.69, value=15.00)  # 输入框

# CRP input
CRP = st.sidebar.number_input("CRP:", min_value=0.09, max_value=5.94, value=5.00)  # 输入框

# hs_TnI input
hs_TnI  = st.sidebar.number_input("hs_TnI:", min_value=0.01, max_value=50.87, value=21.01)  # 输入框

# Creatinine input
Creatinine = st.sidebar.number_input("Creatinine:", min_value=38.00, max_value=1348.00, value=90.00)  # 输入框

# Process the input and make a prediction
feature_values = [CPR_time, Initial_cardiac_rhythm, Bystander_CPR, SOFA_score, sTREM_1, NSE, IL_6, IL_10, CRP, hs_TnI, Creatinine]  # 收集所有输入的特征
features = np.array([feature_values])  # 转换为NumPy数组

#
explainer = shap.TreeExplainer(model)

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为死亡
        advice = (
            f"According to our model, the patient's risk of mortality is {probability:.1f}%. "
        )  # 如果预测为死亡，给出相关建议
    else:  # 如果预测为存活
        advice = (
            f"According to our model, the patient's probability of survival is {probability:.1f}%. "
        )  # 如果预测为存活，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0的概率
        'Class_1': predicted_proba[1]  # 类别1的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['Survival', 'Death'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表

    ## SHAP力图生成 
    shap_values = explainer(features)

    # 使用Matplotlib渲染 
    plt.figure(figsize=(10,6)) 
    shap.plots.waterfall(shap_values[0],  max_display=10)
    st.pyplot(plt.gcf()) 
    
    # 交互式HTML展示 
    st.markdown("Prediction Model with SHAP Visualization")
    shap_html = shap.plots.force(shap_values[0],  matplotlib=False)
    st.components.v1.html(shap_html,  height=400)
