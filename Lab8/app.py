import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="ML Demo - Wine Classification", page_icon="🍷", layout="wide")

st.title("Веб-приложение для демонстрации модели машинного обучения")
st.markdown("### Классификация вин на основе химического состава")
st.markdown("---")

# Боковая панель
st.sidebar.header("Настройки модели")
st.sidebar.subheader("Выбор модели")
st.sidebar.markdown("**Модель:** Random Forest")

st.sidebar.subheader("Гиперпараметры")
n_estimators = st.sidebar.slider("Количество деревьев (n_estimators)", 10, 200, 100, 10)
max_depth = st.sidebar.slider("Максимальная глубина (max_depth)", 1, 20, 10, 1)
test_size = st.sidebar.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)

st.sidebar.markdown("---")
train_button = st.sidebar.button("Обучить модель", type="primary", use_container_width=True)

# Информация о датасете (первая колонка)
st.subheader("Информация о датасете")

wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Количество образцов", X.shape[0])
with col2:
    st.metric("Количество признаков", X.shape[1])
with col3:
    st.metric("Количество классов", len(target_names))

st.write(f"**Целевые классы:** {', '.join(target_names)}")

# ПРИМЕР ДАННЫХ (широкая таблица)
st.subheader("Пример данных (первые 5 строк)")
df_sample = pd.DataFrame(X[:5], columns=feature_names)
df_sample['target'] = y[:5]
df_sample['target_name'] = [target_names[i] for i in y[:5]]
st.dataframe(df_sample, use_container_width=True)

# Обучение и результаты
if train_button:
    with st.spinner("Обучение модели..."):
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Стандартизация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Предсказание
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"Обучение завершено!")
        st.metric("Точность (Accuracy)", f"{accuracy:.4f}")
        
        # CLASSIFICATION REPORT (широкая таблица)
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
        
        # РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ (широкая таблица)
        st.subheader("Распределение предсказаний")
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        dist_df = pd.DataFrame({
            'Класс': target_names,
            'Количество предсказаний': [pred_counts.get(i, 0) for i in range(3)],
            'Процент': [f"{pred_counts.get(i, 0)/len(y_pred)*100:.1f}%" for i in range(3)]
        })
        st.dataframe(dist_df, use_container_width=True)
        
        # Сохраняем в session_state для графиков
        st.session_state['trained'] = True
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['target_names'] = target_names
        st.session_state['feature_names'] = feature_names
        st.session_state['feature_importances'] = model.feature_importances_
        st.session_state['model'] = model
        st.session_state['X_test'] = X_test_scaled

# Графики (если модель обучена)
if st.session_state.get('trained', False):
    st.markdown("---")
    st.subheader("Визуализация результатов")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.write("**Матрица ошибок (Confusion Matrix)**")
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=st.session_state['target_names'],
                    yticklabels=st.session_state['target_names'], ax=ax_cm)
        ax_cm.set_xlabel('Предсказанный класс')
        ax_cm.set_ylabel('Истинный класс')
        st.pyplot(fig_cm)
    
    with col5:
        st.write("**Важность признаков (Feature Importance)**")
        importance_df = pd.DataFrame({
            'Признак': st.session_state['feature_names'],
            'Важность': st.session_state['feature_importances']
        }).sort_values('Важность', ascending=True)
        
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
        ax_fi.barh(importance_df['Признак'], importance_df['Важность'], color='skyblue')
        ax_fi.set_xlabel('Важность')
        ax_fi.set_title('Важность признаков')
        plt.tight_layout()
        st.pyplot(fig_fi)
    
    # ROC-кривые
    st.subheader("📈 ROC-кривые для каждого класса")
    y_bin = label_binarize(st.session_state['y_test'], classes=[0, 1, 2])
    proba = st.session_state['model'].predict_proba(st.session_state['X_test'])
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
    colors = ['blue', 'red', 'green']
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{st.session_state["target_names"][i]} (AUC = {roc_auc:.2f})')
    
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC-кривые для каждого класса')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Лабораторная работа №8 | Студент: Артёмова Дарья Артёмовна | Группа: ИБМ3-64Б"
    "</div>",
    unsafe_allow_html=True
)