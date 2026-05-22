<div align="center">
  <img src="https://img.icons8.com/color/96/000000/dna-helix--v1.png" alt="Logo"/>
  <h1>⚛️ Symmetry-BioPredictor</h1>
  <h3>In-Silico Bio-Designer & ML Predictor</h3>

  <p>
    <a href="https://github.com/tryfuIhaIckme/MOKSPROJECT/stargazers"><img src="https://img.shields.io/badge/Status-Scientific_Prototype-gold?style=for-the-badge&logo=appveyor" alt="Status"></a>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python"></a>
    <a href="https://catboost.ai"><img src="https://img.shields.io/badge/ML-CatBoost-red?style=for-the-badge&logo=scikit-learn" alt="CatBoost"></a>
    <a href="https://streamlit.io"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit"></a>
    <a href="https://docker.com"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker" alt="Docker"></a>
  </p>
</div>

---

## 📖 Описание проекта

**Symmetry-BioPredictor** — это передовая программная платформа для предсказания газопермеации (диффузии) в металлоорганических каркасах (MOF). В отличие от классических моделей, наша система использует **теорию групп** и **симметрийный резонанс** для достижения беспрецедентной точности в дизайне биосенсоров и сепарации газов. 

Система оценивает физическую адекватность предсказаний, анализирует биоактивность металлов-узлов (например, Cu, Zn) и прогнозирует поведение молекул (H2, SF6 и др.) на базе предобученной модели `symmetry_predictor.cbm`.

---

## 🧪 Научная концепция: Симметрийный резонанс

Традиционные методы предсказания диффузии в пористых материалах опираются лишь на изотропные параметры (размеры пор: PLD и LCD). 
Мы интегрировали точечные группы симметрии ($O_h$, $T_d$, $D_{\infty h}$ и др.), проверяя гипотезу **геометрического перекрытия**: 
> Когда элементы симметрии молекулы-гостя (Guest) коррелируют с элементами симметрии порового окна (Host), возникает эффект "облегченной диффузии" (Symmetry Match).

**Ключевой дескриптор (SFF):**
$$SFF = \frac{PLD - \sigma_{gas}}{|G_{host}| - |G_{guest}| + \epsilon}$$

---

## 🚀 Нововведения и Ключевые возможности

✅ **Готовая модель CatBoost (`symmetry_predictor.cbm`)**: Предварительно обученная высокоточная модель, учитывающая новые биоактивные дескрипторы и типы линкеров.  
✅ **KPI Validation (`validate_kpi.py`)**: Встроенный модуль для проверки физической адекватности модели (например, проверка существенной разницы в диффузии между мелкими H2 и крупными SF6 молекулами).  
✅ **Интерактивный дашборд**: Построен на базе Streamlit с поддержкой 3D-визуализации молекул и MOF.  
✅ **Explainable AI (SHAP)**: Прозрачная декомпозиция предсказаний модели. Вы наглядно видите, как геометрия и симметрия влияют на проницаемость газа.  
✅ **Таблица Лидерборда**: Отслеживание и сравнение различных структур MOF прямо в интерфейсе (интеграция метрик качества).  
✅ **Метаданные (`mof_metadata.csv`)**: Расширенная база данных свойств материалов и линкеров.

---

## 🛠 Технический стек

* **Machine Learning**: `CatBoost` (градиентный бустинг), устойчивый к категориальным признакам (симметрии).
* **Interpretability**: `SHAP` (SHapley Additive exPlanations).
* **Chemistry Analysis**: `Pymatgen` (анализ `.cif` структур), `py3Dmol` (3D рендеринг).
* **Frontend UI**: `Streamlit`, `Plotly`.
* **Deployment**: `Docker`, `Docker Compose`.

---

## 📦 Быстрая установка

### Способ 1: С помощью Docker (Рекомендуемый) 🐳

Для максимального удобства и отсутствия конфликтов версий мы добавили поддержку Docker.

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/tryfuIhaIckme/MOKSPROJECT.git
   cd MOKSPROJECT
   ```
2. Запустите сборку и контейнер одной командой:
   ```bash
   docker-compose up --build -d
   ```
3. Откройте приложение в браузере: **`http://localhost:8501`**

### Способ 2: Классическая установка (Virtual Environment) 🐍

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/tryfuIhaIckme/MOKSPROJECT.git
   cd MOKSPROJECT
   ```
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Windows: venv\Scripts\activate
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Запустите интерактивный интерфейс:
   ```bash
   streamlit run app.py
   ```

---

## 🧬 Валидация модели (KPI)

Чтобы убедиться в физической адекватности обученной модели, запустите скрипт проверки KPI:
```bash
python validate_kpi.py
```
Скрипт проверяет отклик модели на биоактивные металлы (например, медь в `CuH3(CO2)3`), извлекает типы линкеров и сверяет разницу в предсказаниях проницаемости между принципиально разными газами (водород и гексафторид серы).

---

<div align="center">
  <i>Разработано с ❤️ для вычислительной химии и дизайна новых материалов</i>
</div>
