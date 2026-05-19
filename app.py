import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ========== Fix Chinese display issues ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
# ================================================

st.set_page_config(page_title="Prediction Model")

with st.spinner("Loading model..."):
    try:
        import shap
        
        try:
            from IPython import get_ipython
        except ImportError:
            pass
        
        # Model path
        model_path = "D:/HuaweiMoveData/Users/wei/Desktop/机器学习代码/机器学习预测模型全流程代码_全网同名_派大珍/2.1Python机器学习建模与评估_非标准化数据/2.训练集构建模型/xgb_model.pkl"
        
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        
        st.success("Model loaded successfully")
        
        def prediction(area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders):
            X = np.array([[area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders]])
            return classifier.predict(X)[0]
        
        st.title("Prediction Model")
        st.markdown("### Please enter the following information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Categorical Variables")
            area = st.selectbox("Area", [0, 1], format_func=lambda x: "Urban" if x == 1 else "Rural")
            Surgery = st.selectbox("Surgery", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Complications = st.selectbox("Complications", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col2:
            st.subheader("Continuous Variables")
            Social_support = st.number_input("Social Support", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            Negative_coping = st.number_input("Negative Coping", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
            anxiety = st.number_input("Anxiety", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
            Sleep_disorders = st.number_input("Sleep Disorders", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
        
        if st.button("Predict", type="primary"):
            result = prediction(area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders)
            
            data_for_prediction = np.array([[area, Surgery, Complications, Social_support, Negative_coping, anxiety, Sleep_disorders]])
            
            if hasattr(classifier, 'predict_proba'):
                proba = classifier.predict_proba(data_for_prediction)
                probability = proba[0][1] * 100
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                if result == 1:
                    st.error(f"Prediction: High Risk")
                    st.warning(f"Risk Probability: {probability:.2f}%")
                else:
                    st.success(f"Prediction: Low Risk")
                    st.info(f"Risk Probability: {probability:.2f}%")
                
                st.progress(int(probability))
            else:
                st.write(f"Prediction: {result}")
            
            # SHAP Feature Importance
            try:
                st.markdown("---")
                st.subheader("Feature Importance Analysis")
                st.markdown("Contribution of each feature to the prediction")
                
                feature_names = ['Area', 'Surgery', 'Complications', 'Social_support', 'Negative_coping', 'Anxiety', 'Sleep_disorders']
                
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(data_for_prediction)
                
                if isinstance(shap_values, list):
                    shap_values_for_class = shap_values[1]
                else:
                    shap_values_for_class = shap_values
                
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Value': shap_values_for_class[0]
                })
                shap_df = shap_df.sort_values('SHAP_Value', key=abs, ascending=False)
                
                # Display table
                st.dataframe(
                    shap_df.style.format({'SHAP_Value': '{:.4f}'}).bar(subset=['SHAP_Value'], color='#ff6b6b', align='mid'),
                    use_container_width=True
                )
                
                # Draw bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                features_sorted = shap_df['Feature'].tolist()
                shap_values_sorted = shap_df['SHAP_Value'].tolist()
                
                colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in shap_values_sorted]
                
                y_pos = range(len(features_sorted))
                bars = ax.barh(y_pos, shap_values_sorted, color=colors, height=0.6)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features_sorted, fontsize=10)
                
                ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
                
                ax.set_xlabel('SHAP Value', fontsize=11)
                ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
                
                max_val = max(abs(max(shap_values_sorted)), abs(min(shap_values_sorted)), 0.1)
                for i, (bar, val) in enumerate(zip(bars, shap_values_sorted)):
                    if val > 0:
                        ax.text(val + 0.01 * max_val, i, f'{val:.3f}', va='center', ha='left', fontsize=9)
                    else:
                        ax.text(val - 0.01 * max_val, i, f'{val:.3f}', va='center', ha='right', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("Red = pushes toward High Risk | Blue = pushes toward Low Risk")
                st.caption("Larger absolute SHAP value = greater impact on prediction")
                
            except Exception as e:
                st.warning(f"Feature importance analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Sidebar instructions
        with st.sidebar:
            st.markdown("## Instructions")
            st.markdown("""
            ### 1. Categorical Variables
            - **Area**: 0=Rural, 1=Urban
            - **Surgery**: 0=No, 1=Yes
            - **Complications**: 0=No, 1=Yes
            
            ### 2. Continuous Variables
            - **Social Support**: Higher score indicates better support
            - **Negative Coping**: Higher score indicates more negative coping
            - **Anxiety**: Higher score indicates more severe anxiety
            - **Sleep Disorders**: Higher score indicates more severe sleep problems
            
            ### 3. Prediction Result
            - **Low Risk**: outcome = 0
            - **High Risk**: outcome = 1
            """)
        
        # Model information
        with st.expander("Model Information"):
            st.write(f"Model Type: {type(classifier).__name__}")
            if hasattr(classifier, 'predict_proba'):
                st.write("Model supports probability prediction")
            st.write("Features used: Area, Surgery, Complications, Social Support, Negative Coping, Anxiety, Sleep Disorders")
    
    except Exception as e:
        st.error(f"Failed to load: {str(e)}")
        st.info("Please check if the model file path is correct")
            if hasattr(classifier, 'predict_proba'):
                st.write("✅ 模型支持概率预测")
            
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
