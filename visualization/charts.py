import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_heatmap(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    z = cm.tolist()
    
    # Renkli ısı haritası oluştur
    fig = ff.create_annotated_heatmap(z, x=labels, y=labels, colorscale='Viridis', showscale=True)
    
    fig.update_layout(
        title_text='<b>Hata Matrisi (Tahmin Başarısı)</b>',
        xaxis_title="Yapay Zekanın Tahmini",
        yaxis_title="Gerçek Sağlık Durumu",
        height=500
    )
    # X ekseni yazıları altta dursun
    fig['layout']['xaxis']['side'] = 'bottom'
    return fig

def plot_feature_importance_bar(model, feature_names, tr_mapping=None):
    """
    tr_mapping: İngilizce sütun isimlerini Türkçe'ye çeviren sözlük
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Eğer çeviri sözlüğü varsa isimleri Türkçe yap, yoksa olduğu gibi bırak
        if tr_mapping:
            translated_names = [tr_mapping.get(name, name) for name in feature_names]
        else:
            translated_names = feature_names
        
        feature_df = pd.DataFrame({
            'Özellik': translated_names,
            'Etki Derecesi': importances
        }).sort_values(by='Etki Derecesi', ascending=True)
        
        fig = px.bar(feature_df, x='Etki Derecesi', y='Özellik', orientation='h', 
                     title="<b>Özellik Önem Düzeyleri (Model Neye Bakıyor?)</b>",
                     color='Etki Derecesi', color_continuous_scale='Bluered')
        
        fig.update_layout(height=700, xaxis_title="Önem Derecesi", yaxis_title="Klinik Bulgular")
        return fig
    return None