import streamlit as st
import numpy as np
import pickle
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="预测模型")

with st.spinner("正在加载模型..."):
    try:
        import shap
        
        # 尝试导入 IPython
        try:
            from IPython import get_ipython
        except ImportError:
            pass
        
        # 模型路径
        model_path = "D:/HuaweiMoveData/Users/wei/Desktop/机器学习代码/机器学习预测模型全流程代码_全网同名_派大珍/2.1Python机器学习建模与评估_非标准化数据/2.训练集构建模型/xgb_model.pkl"
        
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        
        st.success("模型加载成功")
        
        def prediction(area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders):
            X = np.array([[area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders]])
            return classifier.predict(X)[0]
        
        st.title("预测模型")
        st.markdown("### 请输入以下信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("分类变量")
            area = st.selectbox("地区", [0, 1], format_func=lambda x: "城市" if x == 1 else "农村")
            Surgery = st.selectbox("手术", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
            Complications = st.selectbox("并发症", [0, 1], format_func=lambda x: "有" if x == 1 else "无")
        
        with col2:
            st.subheader("连续变量")
            Social_support = st.number_input("社会支持度", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            Negative_coping = st.number_input("消极应对方式", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
            anxiety = st.number_input("焦虑评分", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
            Sleep_disorders = st.number_input("睡眠障碍评分", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
        
        if st.button("开始预测", type="primary"):
            result = prediction(area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders)
            
            data_for_prediction = np.array([[area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders]])
            
            if hasattr(classifier, 'predict_proba'):
                proba = classifier.predict_proba(data_for_prediction)
                probability = proba[0][1] * 100
                
                if result == 1:
                    st.error(f"⚠️ 预测结果：高风险")
                    st.warning(f"风险概率：{probability:.2f}%")
                else:
                    st.success(f"✅ 预测结果：低风险")
                    st.info(f"风险概率：{probability:.2f}%")
                
                st.progress(int(probability))
            
            # SHAP解释
            try:
                st.markdown("---")
                st.subheader("特征重要性分析")
                
                feature_names = ['地区', '手术', '并发症', '社会支持度', '消极应对', '焦虑', '睡眠障碍']
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(data_for_prediction)
                
                if isinstance(shap_values, list):
                    shap_values_for_class = shap_values[1]
                else:
                    shap_values_for_class = shap_values
                
                # 创建条形图
                shap_df = pd.DataFrame({
                    '特征': feature_names,
                    'SHAP值': shap_values_for_class[0]
                })
                shap_df = shap_df.sort_values('SHAP值', key=abs, ascending=False)
                
                st.write("各特征对预测结果的贡献：")
                
                # 颜色编码的条形图
                colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in shap_df['SHAP值']]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(shap_df['特征'], shap_df['SHAP值'], color=colors)
                ax.axvline(x=0, color='black', linewidth=0.5)
                ax.set_xlabel('SHAP值（对预测的影响程度）')
                ax.set_title('特征贡献度')
                st.pyplot(fig)
                
                st.caption("红色 = 促使预测为高风险，蓝色 = 促使预测为低风险")
                
            except Exception as e:
                st.warning(f"SHAP解释生成失败：{str(e)}")
        
        # 侧边栏
        with st.sidebar:
            st.markdown("## 📖 使用说明")
            st.markdown("""
            ### 1. 分类变量
            - **地区**：0=农村，1=城市
            - **手术**：0=否，1=是
            - **并发症**：0=无，1=有
            
            ### 2. 连续变量
            - **社会支持度**：分值越高表示社会支持越好
            - **消极应对方式**：分值越高表示消极应对越多
            - **焦虑评分**：分值越高表示焦虑越严重
            - **睡眠障碍评分**：分值越高表示睡眠问题越严重
            
            ### 3. 预测结果说明
            - **低风险**：outcome = 0
            - **高风险**：outcome = 1
            
            ---
            
            ### ⚠️ 注意事项
            - 本预测结果仅供参考
            - 不能替代专业医疗建议
            - 请确保输入数据准确
            """)
        
        # 显示模型信息
        with st.expander("模型信息"):
            st.write(f"模型类型：{type(classifier)}")
            if hasattr(classifier, 'predict_proba'):
                st.write("✅ 模型支持概率预测")
            
    except Exception as e:
        st.error(f"加载失败：{str(e)}")