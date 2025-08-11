import streamlit as st
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Здесь мы будем хранить "эталонные" категории
        self.categories_ = {}

    def fit(self, X, y=None):
        # На этапе fit мы "запоминаем" все возможные категории для каждой колонки
        for col in ['sex', 'smoker', 'region']:
            self.categories_[col] = X[col].astype('category').cat.categories
        return self

    def transform(self, X, y=None):
        df = X.copy()
        
        # Применяем тип category с "запомненными" категориями
        for col, cats in self.categories_.items():
            df[col] = pd.Categorical(df[col], categories=cats)
            
        # Теперь get_dummies всегда будет создавать одинаковый набор колонок
        df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)
        
        # Новые признаки
        # Добавляем .get('smoker_yes', 0), чтобы код не падал, если такой колонки нет
        # (хотя теперь она должна быть всегда)
        df['age_smoker'] = df['age'] * df.get('smoker_yes', 0)
        df['bmi_smoker'] = df['bmi'] * df.get('smoker_yes', 0)
        
        return df


# --- Словари для "перевода" ---
SEX_MAP = {'Женский': 'female', 'Мужской': 'male'}
SMOKER_MAP = {'Нет': 'no', 'Да': 'yes'}
REGION_MAP = {
    'юго-запад': 'southwest',
    'юго-восток': 'southeast',
    'северо-запад': 'northwest',
    'северо-восток': 'northeast'
}

def main():
    st.title("Прогнозирование стоимости медицинской страховки в США")

    # --- Сбор данных от пользователя ---
    age = st.number_input("Введите возраст", min_value=18, max_value=150, value=25)
    sex_ru = st.selectbox("Выберите пол", ['Женский', 'Мужской'])
    bmi = st.number_input("Введите ИМТ", min_value=15.0, max_value=60.0, value=28.5)
    children = st.number_input("Сколько у вас детей?", min_value=0, value=0)
    smoker_ru = st.selectbox('Вы курите?', ['Нет', 'Да'])
    region_ru = st.selectbox('В каком регионе США вы живете?', ['юго-запад', "юго-восток", "северо-запад", "северо-восток"])

    # --- Кнопка и логика предсказания ---
    if st.button("Предсказать стоимость страховки"):
        
        # --- Преобразование данных в формат, понятный модели ---
        sex_en = SEX_MAP[sex_ru]
        smoker_en = SMOKER_MAP[smoker_ru]
        region_en = REGION_MAP[region_ru]

        # Создаем DataFrame с сырыми данными на английском
        data = pd.DataFrame({
            'age': [age],
            'sex': [sex_en],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_en],
            'region': [region_en]
        })

        try:
            # Загружаем пайплайн
            pipeline = joblib.load('pipeline.pkl') 
            
            # Делаем предсказание
            log_prediction = pipeline.predict(data)
            prediction = np.expm1(log_prediction)
            
            # Выводим результат
            st.success(f"Предсказанная стоимость страховки: ${prediction[0]:,.2f}")
        
        except FileNotFoundError:
            st.error("Файл модели 'insurance_pipeline_final.pkl' не найден. Убедитесь, что он находится в той же папке.")
        except Exception as e:
            st.error(f"Произошла ошибка при предсказании: {e}")


if __name__ == "__main__":
    main()