import streamlit as st

# --- 2. CSS İLE ZORLA GENİŞLETME (Paddingleri alma) ---
st.markdown(
    """
    <style>
        /* 1. Sayfanın ana bloğundaki boşlukları al ve %100 yap */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }

        /* 2. Sidebar genişliğini sabitle */
        [data-testid="stSidebar"] {
            min-width: 350px !important;
            max-width: 350px !important;
        }
        
        /* 3. DataFrame ve Grafikleri tam boy yap */
        .stDataFrame, .stPlotlyChart {
            width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

import pandas as pd
import plotly.express as px
import time
import numpy as np
import joblib
import os
from data_pipeline.loader import load_data
from data_pipeline.preprocessing import clean_data, scale_features, handle_missing_values
from model_factory.trainers import get_model
from evaluation.comparisons import train_and_evaluate
from visualization.charts import plot_confusion_matrix_heatmap, plot_feature_importance_bar
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


# --- 2. TÜRKÇE SÖZLÜK ---
tr_labels = {
    'baseline value': 'Temel Kalp Hızı (Normal: 110-160)',
    'accelerations': 'Hızlanmalar (İyi: > 0.003)',
    'fetal_movement': 'Bebek Hareket Sayısı',
    'uterine_contractions': 'Rahim Kasılmaları (0.0 - 0.015)',
    'light_decelerations': 'Hafif Yavaşlamalar',
    'severe_decelerations': 'Şiddetli Yavaşlamalar',
    'prolongued_decelerations': 'Uzun Yavaşlamalar (Normal: 0.0)',
    'abnormal_short_term_variability': 'Anormal Kısa Dönem Değişkenlik (Normal: < 60)',
    'mean_value_of_short_term_variability': 'Kısa Dönem Değişkenlik Ortalaması',
    'percentage_of_time_with_abnormal_long_term_variability': 'Anormal Uzun Dönem Değişkenlik %',
    'mean_value_of_long_term_variability': 'Uzun Dönem Değişkenlik Ortalaması',
    'histogram_width': 'Histogram Genişliği',
    'histogram_min': 'Histogram Min Değer',
    'histogram_max': 'Histogram Max Değer',
    'histogram_number_of_peaks': 'Histogram Tepe Noktası',
    'histogram_number_of_zeroes': 'Histogram Sıfır Sayısı',
    'histogram_mode': 'Histogram Mod (Tepe Değer)',
    'histogram_mean': 'Histogram Ortalama',
    'histogram_median': 'Histogram Medyan',
    'histogram_variance': 'Histogram Varyansı',
    'histogram_tendency': 'Histogram Eğilimi'
}

# --- YARDIMCI FONKSİYON: GAUGE CHART ---
def create_gauge_chart(title, value, color_code):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18}},
        number = {'suffix': "%", 'font': {'size': 24}}, 
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_code}, 
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 100], 'color': "#f0f0f0"}
            ],
            'shape': "angular", 
        }
    ))
    fig.update_layout(autosize=True, height=180, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#333"})
    return fig

# --- YARDIMCI FONKSİYONLAR ---
def try_load_model(silent=False):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "champion_model.pkl")
    
    if os.path.exists(file_path):
        try:
            model_data = joblib.load(file_path)
            st.session_state['best_model'] = model_data['model']
            st.session_state['scaler'] = model_data['scaler']
            st.session_state['best_model_name'] = model_data['name']
            st.session_state['best_accuracy'] = model_data['accuracy']
            if 'X_test' in model_data: st.session_state['last_X_test'] = model_data['X_test']
            if 'y_test' in model_data: st.session_state['last_y_test'] = model_data['y_test']
            if 'feature_names' in model_data: st.session_state['feature_names'] = model_data['feature_names']
            if 'results_df' in model_data: st.session_state['last_results'] = model_data['results_df']
            
            if not silent: return True, f"{model_data['name']} ({file_path})"
            return True, "Oto Yükleme Başarılı"
        except Exception as e: return False, str(e)
    return False, "Dosya bulunamadı."

@st.cache_data
def get_data():
    df = load_data("data/fetal_health.csv")
    return clean_data(df)

# --- 3. ANA FONKSİYON ---
def run_ui():
    keys_to_init = {
        'best_model': None, 'best_model_name': "", 'best_accuracy': 0.0,
        'scaler': None, 'last_results': None, 'last_X_test': None,
        'last_y_test': None, 'feature_names': None, 'auto_loaded': False
    }
    for key, val in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if not st.session_state['auto_loaded']:
        success, msg = try_load_model(silent=True)
        if success:
            st.toast(f"🚀 Hazır Model Otomatik Yüklendi: {st.session_state['best_model_name']}", icon="✅")
        st.session_state['auto_loaded'] = True
    
    st.title("🏥 Fetal Sağlık - Yapay Zeka Arenası")
    
    st.sidebar.header("Navigasyon")
    page = st.sidebar.selectbox("Modül Seçiniz", ["1. Veri Analizi", "2. Model Kıyaslama (Toplu Eğitim)", "3. Canlı Tahmin"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Model Durumu")
    
    if st.session_state.get('best_model'):
        st.sidebar.success(f"✅ AKTİF: **{st.session_state['best_model_name']}**")
        st.sidebar.info(f"Başarı Skoru: **%{st.session_state['best_accuracy']*100:.2f}**")
        if st.sidebar.button("🔄 Dosyadan Tekrar Yükle"):
            success, msg = try_load_model()
            if success:
                st.sidebar.success("Yüklendi!")
                time.sleep(1)
                st.rerun()
    else:
        st.sidebar.warning("Henüz model yok.")
        if st.sidebar.button("📂 Kayıtlı Modeli Ara ve Yükle"):
            success, msg = try_load_model()
            if success: st.rerun()

    df = get_data()
    
    if page == "1. Veri Analizi":
        show_data_analysis(df)
    elif page == "2. Model Kıyaslama (Toplu Eğitim)":
        show_model_comparison(df)
    elif page == "3. Canlı Tahmin":
        show_prediction(df)

def show_data_analysis(df):
    st.header("📊 Veri Seti Analizi")
    show_all = st.checkbox("Tüm verileri göster", value=False)
    
    if show_all:
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df.head(7), use_container_width=True)
        st.caption("Varsayılan olarak ilk 7 satır gösteriliyor.")

    col1, col2 = st.columns(2)
    col1.info(f"Toplam Veri Sayısı: {df.shape[0]}")
    col2.info(f"Toplam Özellik (Sütun): {df.shape[1]}")
    
    fig_target = px.histogram(df, x="fetal_health", color="fetal_health", title="Sınıf Dağılımı (1: Normal, 2: Şüpheli, 3: Patolojik)")
    st.plotly_chart(fig_target, use_container_width=True)

def show_model_comparison(df):
    st.header("🏆 Algoritma Kıyaslama Arenası")
    
    if st.session_state.get('best_model') is not None:
        st.success(f"📂 Hafızadaki Şampiyon: **{st.session_state['best_model_name']}** (Başarı: %{st.session_state['best_accuracy']*100:.2f})")
        col_save, col_info = st.columns([1, 2])
        with col_save:
            if st.button("💾 BU MODELİ VE GRAFİKLERİ DİSKE KAYDET", type="primary", use_container_width=True):
                try:
                    current_dir = os.getcwd()
                    file_path = os.path.join(current_dir, "champion_model.pkl")
                    save_package = {
                        'model': st.session_state['best_model'],
                        'scaler': st.session_state['scaler'],
                        'name': st.session_state['best_model_name'],
                        'accuracy': st.session_state['best_accuracy'],
                        'X_test': st.session_state.get('last_X_test'),
                        'y_test': st.session_state.get('last_y_test'),
                        'feature_names': st.session_state.get('feature_names'),
                        'results_df': st.session_state.get('last_results')
                    }
                    joblib.dump(save_package, file_path)
                    st.balloons()
                    st.success(f"✅ Kayıt Başarılı!")
                except Exception as e: st.error(f"Hata: {e}")

    if st.session_state.get('last_results') is not None:
        st.markdown("---")
        st.subheader("📊 Sonuç Tablosu")
        results_df = st.session_state['last_results']
        col_table, col_graph = st.columns([1, 1])
        with col_table:
            st.dataframe(results_df.style.background_gradient(subset=["Accuracy"], cmap="Greens").format({"Accuracy": "{:.2%}", "F1": "{:.2%}", "Time": "{:.4f}"}), use_container_width=True)
        with col_graph:
            fig = px.bar(results_df, x="Model", y="Accuracy", color="Model", title="Model Başarı Karşılaştırması")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.get('best_model') and st.session_state.get('last_X_test') is not None:
            st.markdown("---")
            st.subheader(f"🧠 {st.session_state['best_model_name']} - Detaylı Performans Analizi")
            tab1, tab2 = st.tabs(["Hata Matrisi", "Özellik Önemi"])
            with tab1:
                y_pred = st.session_state['best_model'].predict(st.session_state['last_X_test'])
                fig_cm = plot_confusion_matrix_heatmap(st.session_state['last_y_test'], y_pred, labels=['Normal', 'Şüpheli', 'Patolojik'])
                st.plotly_chart(fig_cm, use_container_width=True)
            with tab2:
                fig_feat = plot_feature_importance_bar(st.session_state['best_model'], st.session_state['feature_names'], tr_labels)
                if fig_feat: st.plotly_chart(fig_feat, use_container_width=True)
                else: st.info("Bu model için özellik önem grafiği desteklenmiyor.")

    st.markdown("---")
    st.subheader("Yeni Turnuva Başlat")
    col_set1, col_set2, col_set3 = st.columns(3)
    n_runs = col_set1.slider("Tur Sayısı", 1, 100, 5) 
    test_size = col_set2.slider("Test Oranı (%)", 10, 50, 20) / 100
    missing_strategy = col_set3.radio("Eksik Veri Stratejisi", ["mean", "median", "drop"], index=1)
    
    models_to_test = ["Random Forest", "SVM", "XGBoost", "Logistic Regression", "Decision Tree"]
    selected_models = st.multiselect("Modelleri Seç", models_to_test, default=models_to_test)

    if st.button("🔥 TURNUVAYI BAŞLAT", use_container_width=True):
        if not selected_models:
            st.error("Model seçmelisin!")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        
        X = df.drop('fetal_health', axis=1)
        y = df['fetal_health'] - 1 
        feature_names_list = X.columns.tolist() 
        X = handle_missing_values(X, strategy=missing_strategy)
        
        global_best_acc = 0
        global_best_model = None
        global_best_name = ""
        global_scaler = None
        final_X_test_scaled = None
        final_y_test = None

        total_steps = len(selected_models) * n_runs
        current_step = 0

        for model_name in selected_models:
            accuracies = []
            f1_scores = []
            times = []
            temp_X_test_scaled = None
            temp_y_test = None
            temp_model = None
            temp_scaler = None

            for i in range(n_runs):
                status_text.text(f"⏳ {model_name} eğitiliyor ({i+1}/{n_runs})...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                X_train_scaled, scaler = scale_features(X_train, method="Standard")
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
                
                temp_X_test_scaled = X_test_scaled
                temp_y_test = y_test
                temp_scaler = scaler
                
                model = get_model(model_name)
                res = train_and_evaluate(model, X_train_scaled, X_test_scaled, y_train, y_test)
                
                accuracies.append(res['Accuracy'])
                f1_scores.append(res['F1 Score'])
                times.append(res['Training Time (sec)'])
                temp_model = res['Model']
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            avg_acc = np.mean(accuracies)
            results_list.append({"Model": model_name, "Accuracy": avg_acc, "F1": np.mean(f1_scores), "Time": np.mean(times), "Tur Sayısı": n_runs})
            
            if avg_acc > global_best_acc:
                global_best_acc = avg_acc
                global_best_model = temp_model
                global_best_name = model_name
                global_scaler = temp_scaler
                final_X_test_scaled = temp_X_test_scaled
                final_y_test = temp_y_test

        progress_bar.empty()
        status_text.success("Turnuva Tamamlandı!")
        res_df = pd.DataFrame(results_list).sort_values(by="Accuracy", ascending=False)
        
        st.session_state['best_model'] = global_best_model
        st.session_state['best_model_name'] = global_best_name
        st.session_state['best_accuracy'] = global_best_acc
        st.session_state['scaler'] = global_scaler
        st.session_state['last_results'] = res_df 
        st.session_state['last_X_test'] = final_X_test_scaled
        st.session_state['last_y_test'] = final_y_test
        st.session_state['feature_names'] = feature_names_list
        st.rerun()

def show_prediction(df):
    st.header("🩺 Canlı Tahmin")
    if st.session_state.get('best_model') is None:
        st.error("⚠️ Model Yüklü Değil!")
        return

    model = st.session_state['best_model']
    scaler = st.session_state['scaler']
    st.success(f"Aktif Model: {st.session_state['best_model_name']}")

    class_means = df.groupby('fetal_health').mean()
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("✅ Sağlıklı Değerleri Doldur", use_container_width=True):
            healthy_data = class_means.loc[1.0]
            for col in healthy_data.index: st.session_state[f"input_{col}"] = healthy_data[col]
            st.rerun()

    with col_demo2:
        if st.button("🚨 Hasta (Patolojik) Değerleri Doldur"):
            sick_data = class_means.loc[3.0]
            for col in sick_data.index:
                st.session_state[f"input_{col}"] = sick_data[col]
            
            st.session_state["input_abnormal_short_term_variability"] = 78.0
            st.session_state["input_prolongued_decelerations"] = 0.005 
            st.session_state["input_accelerations"] = 0.000
            st.session_state["input_baseline value"] = 100.0
            st.session_state["input_histogram_mean"] = 100.0
            st.session_state["input_histogram_median"] = 100.0
            st.session_state["input_histogram_mode"] = 100.0
            st.session_state["input_histogram_min"] = 90.0
            st.session_state["input_histogram_max"] = 110.0
            st.session_state["input_histogram_width"] = 20.0
            st.rerun()
            
    st.markdown("---")
    use_sync = st.checkbox("🔗 Otomatik Tutarlılık (Histogram Matematiksel Eşitleme)", value=False)
    input_values = {}
    original_columns = df.drop('fetal_health', axis=1).columns
    main_cols = ['baseline value', 'accelerations', 'uterine_contractions', 'prolongued_decelerations', 'abnormal_short_term_variability']
    other_cols = [c for c in original_columns if c not in main_cols]
    
    col1, col2 = st.columns(2)
    with col1:
        col_name = main_cols[0] 
        label = tr_labels.get(col_name, col_name)
        def_val = st.session_state.get(f"input_{col_name}", float(df[col_name].median()))
        new_baseline = st.number_input(label, value=float(def_val), key=f"input_{col_name}")
        input_values[col_name] = new_baseline
        if use_sync:
            input_values['histogram_mean'] = new_baseline
            input_values['histogram_median'] = new_baseline
            input_values['histogram_mode'] = new_baseline
            input_values['histogram_min'] = max(0, new_baseline - 10)
            input_values['histogram_max'] = new_baseline + 10
            new_width = input_values['histogram_max'] - input_values['histogram_min']
            input_values['histogram_width'] = new_width
            st.session_state['input_histogram_mean'] = input_values['histogram_mean']
            st.session_state['input_histogram_median'] = input_values['histogram_median']
            st.session_state['input_histogram_mode'] = input_values['histogram_mode']
            st.session_state['input_histogram_min'] = input_values['histogram_min']
            st.session_state['input_histogram_max'] = input_values['histogram_max']
            st.session_state['input_histogram_width'] = input_values['histogram_width']
    
        for col_name in main_cols[1:3]: 
             label = tr_labels.get(col_name, col_name)
             def_val = st.session_state.get(f"input_{col_name}", float(df[col_name].median()))
             input_values[col_name] = st.number_input(label, value=float(def_val), format="%.4f", key=f"input_{col_name}")
    with col2:
        for col_name in main_cols[3:]:
             label = tr_labels.get(col_name, col_name)
             def_val = st.session_state.get(f"input_{col_name}", float(df[col_name].median()))
             input_values[col_name] = st.number_input(label, value=float(def_val), format="%.4f", key=f"input_{col_name}")

    with st.expander("🔬 Detaylı Veriler (Histogram, Varyans vb.)", expanded=True):
        cols = st.columns(4)
        for i, col_name in enumerate(other_cols):
            with cols[i % 4]:
                label = tr_labels.get(col_name, col_name)
                if use_sync and col_name in ['histogram_mean', 'histogram_median', 'histogram_mode', 'histogram_min', 'histogram_max', 'histogram_width']:
                     val_to_show = input_values.get(col_name, 0)
                     st.number_input(label, value=float(val_to_show), key=f"disp_{col_name}", disabled=True)
                     input_values[col_name] = val_to_show
                else:
                    def_val = st.session_state.get(f"input_{col_name}", float(df[col_name].median()))
                    val = st.number_input(label, value=float(def_val), key=f"input_{col_name}")
                    input_values[col_name] = val

    if st.button("🔍 TAHMİN ET", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_values])
        input_df = input_df[original_columns]
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=original_columns)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("---")
        if pred == 0.0: st.success("🟢 SONUÇ: NORMAL")
        elif pred == 1.0: st.warning("🟡 SONUÇ: ŞÜPHELİ")
        else: st.error("🔴 SONUÇ: PATOLOJİK (RİSKLİ)")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            
            # --- OLASILIK DAĞILIMI (GAUGE CHARTS) ---
            st.markdown("### 🎯 Olasılık Dağılımı (Güven Oranı)")
            col_g1, col_g2, col_g3 = st.columns(3)
            
            with col_g1:
                st.plotly_chart(create_gauge_chart("🟢 Normal", probs[0]*100, "#2ecc71"), use_container_width=True)
            with col_g2:
                st.plotly_chart(create_gauge_chart("🟡 Şüpheli", probs[1]*100, "#f1c40f"), use_container_width=True)
            with col_g3:
                st.plotly_chart(create_gauge_chart("🔴 Patolojik", probs[2]*100, "#e74c3c"), use_container_width=True)
        
        # --- KARAR ANALİZİ: NEDEN BU SONUÇ ÇIKTI? (YENİLENEN KISIM) ---
        st.markdown("---")
        st.subheader("🕵️ Karar Analizi: Neden Bu Sonuç Çıktı?")
        
        # Sadece Tree-based modellerde (XGBoost, RandomForest vb.) çalışır
        if hasattr(model, "feature_importances_"):
            import numpy as np
            
            # 1. Modelin genel özellik önemleri
            importances = model.feature_importances_
            
            # 2. Girilen hastanın değerlerinin 'Normalden Sapma' miktarı
            # input_scaled içinde 0 demek 'tam ortalama' demek.
            # Mutlak değer alıyoruz ki eksi veya artı yönde aşırı sapmaları yakalayalım.
            input_impact = np.abs(input_scaled.values[0])
            
            # 3. HİBRİT SKOR: (Önem x Sapma)
            # Hem model için önemli olacak HEM DE hasta bu konuda uç değerde olacak.
            contribution = importances * input_impact
            
            # Eğer tüm değerler ortalamaysa (0 ise) hata vermesin diye minik bir sayı ekle
            if contribution.sum() == 0: contribution += 1e-9
                
            # 4. Yüzdeye Çevirme (%30 buradan, %20 şuradan...)
            contribution_pct = (contribution / contribution.sum()) * 100
            
            # 5. DataFrame Hazırlığı
            reason_df = pd.DataFrame({
                'Özellik': original_columns,
                'Etki Yüzdesi': contribution_pct,
                'Ham Önem': importances,
                'Hasta Değeri (Scaled)': input_impact
            })
            
            # Türkçe İsimlendirme
            reason_df['Özellik İsmi'] = reason_df['Özellik'].apply(lambda x: tr_labels.get(x, x))
            
            # En etkili 5 sebebi al
            reason_df = reason_df.sort_values(by='Etki Yüzdesi', ascending=False).head(5)
            
            col_reason1, col_reason2 = st.columns([1, 1])
            
            with col_reason1:
                # PASTA GRAFİĞİ (DONUT)
                fig_pie = px.pie(
                    reason_df, 
                    values='Etki Yüzdesi', 
                    names='Özellik İsmi',
                    title=f"Bu Kararı Etkileyen En Büyük 5 Faktör",
                    hole=0.4, # Ortası delik olsun (Donut)
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0), height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_reason2:
                # YANINA AÇIKLAMA YAZALIM
                top_feature = reason_df.iloc[0]['Özellik İsmi']
                top_pct = reason_df.iloc[0]['Etki Yüzdesi']
                
                st.info(f"💡 **Yapay Zeka Diyor ki:**\n\n"
                        f"Verdiğim kararda en büyük etken **%{top_pct:.1f}** oranıyla\n"
                        f"**'{top_feature}'** verisindeki anormalliktir.\n\n"
                        f"Bu hastanın bu değeri, normal standartların dışına çıkmış ve kararı tetiklemiş.")
                
                st.dataframe(
                    reason_df[['Özellik İsmi', 'Etki Yüzdesi']].style.format({'Etki Yüzdesi': '%{:.1f}'}),
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.warning("Bu model türü (Örn: SVM) detaylı etki analizini desteklemiyor.")