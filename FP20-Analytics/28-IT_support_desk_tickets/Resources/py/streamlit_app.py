import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import difflib
from wordcloud import WordCloud
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from transformers import pipeline

# --- Procesamiento de datos (copiado de analysis.py) ---
def normalize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def add_resolution_time_and_custom_tag(df):
    cols = df.columns
    date_col = next((c for c in cols if "date" in c and "resolution" not in c), None)
    resolution_col = next((c for c in cols if "resolution_date" in c), None)
    if date_col and resolution_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[resolution_col] = pd.to_datetime(df[resolution_col], errors='coerce')
        df['resolution_time_days'] = (df[resolution_col] - df[date_col]).dt.days
    else:
        df['resolution_time_days'] = np.nan
    def get_custom_tag(row):
        primary = str(row.get('primary_tag', '')).lower()
        secondary = str(row.get('secondary_tag', '')).lower()
        for tag in [secondary, primary]:
            if any(x in tag for x in ['bug', 'technical', 'security']):
                if 'bug' in tag:
                    return 'Bug'
                if 'technical' in tag:
                    return 'Technical'
                if 'security' in tag:
                    return 'Security'
        return row.get('primary_tag', '')
    df['custom_tag'] = df.apply(get_custom_tag, axis=1)
    return df

def load_and_prepare_data():
    file_path = r"C:/Users/erlin/CursorProjects/power-bi/FP20-Analytics/28-IT_support_desk_tickets/IT_Support_Ticket_Desk_English.xlsx"
    df = pd.read_excel(file_path)
    df = normalize_columns(df)
    df = add_resolution_time_and_custom_tag(df)
    date_col = next((c for c in df.columns if c.startswith('date') and 'resolution' not in c), None)
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    df = df[df['month'].notnull()]
    return df, date_col

# --- Streamlit App ---
st.set_page_config(page_title="IT Support Desk Tickets - Streamlit Analytics", layout="wide")
df, date_col = load_and_prepare_data()

# --- Custom Tag mapping for clarity in heatmap ---
custom_tag_map = {
    'upgrade': 'Sales',
    'tech support': 'Technical',
    'critical data loss': 'Bug',
    'api': 'Technical',
    'features': 'Feature',
    'campaigns': 'campaign',
    'socialmedia': 'marketing',
    'breach': 'security',
    'refund': 'return',
    'social media': 'marketing',
    'pricing': 'sales',
    'inquiry': 'customer',
    'sales': 'sales',
    'notification': 'feature',
    'brand': 'marketing',
    'social': 'marketing',
    'website': 'feature',
    'ad': 'feature',
    'shipment': 'sales',
    'digitalmarketing': 'marketing',
    'investment': 'sales',
    'brand growth': 'marketing',
    'bug': 'bug',
    'lead': 'customer',
    'access': 'customer',
    'digital campaign': 'marketing',
    'server': 'security',
    'billing,product,error,data analytics,breach': 'security',
    'digital,campaign,performance,issue,resolution,feedback': 'security',
    'business': 'security',
    'malware': 'security',
    'secure': 'security',
    'feature': 'feature',
    'digital,campaign,performance,issue,resolution,urgency,communicat': 'bug',
    'customer,sales,strategy,online presence,method,website,interaction': 'marketing',
    'slow,growth,strategy,improvement,issue,solution,performance': 'marketing',
    'gaming': 'marketing',
    'issue': 'bug',
    'urgent': 'security',
    'device': 'security',
    'disappearance': 'security',
}
df['custom_tag'] = df['custom_tag'].astype(str).str.strip().str.lower().replace(custom_tag_map)

st.title("IT Support Desk Tickets - Streamlit Analytics")

# Normalizar columna priority y preparar tabla resumen
if not isinstance(df['priority'], pd.Series):
    df['priority'] = pd.Series(df['priority'])
priority_counts = pd.Series(df['priority']).astype(str).str.strip().str.lower().value_counts().reset_index()
priority_counts.columns = ['Priority', 'Count']
df['priority'] = pd.Series(df['priority']).astype(str).str.strip().str.lower()

# --- Análisis exploratorio de resolution_time_days ---
st.header('Análisis de resolution_time_days')
res_desc = pd.Series(df['resolution_time_days']).describe()
res_unique = list(pd.Series(df['resolution_time_days']).dropna().unique())
res_by_priority = df.groupby('priority')['resolution_time_days'].mean().reset_index()
res_by_queue = df.groupby('queue')['resolution_time_days'].mean().reset_index()

st.write('**Resumen estadístico:**')
st.dataframe(res_desc)
st.write('**Valores únicos:**', res_unique)
st.write('**Promedio por priority:**')
st.dataframe(res_by_priority)
st.write('**Promedio por queue:**')
st.dataframe(res_by_queue)

# Chequeo de posible error: si todos los valores son iguales o NaN
if len(res_unique) <= 1:
    st.warning('⚠️ Todos los valores de resolution_time_days son iguales o NaN. Puede haber un error en el cálculo de la columna. Revisando fechas...')
    # Revisar columnas de fecha
    st.write('**Primeros valores de las columnas de fecha:**')
    date_cols = [c for c in df.columns if 'date' in c]
    st.dataframe(df.loc[:, date_cols].head(10))
    # Intentar recalcular resolution_time_days
    date_col = next((c for c in df.columns if c.startswith('date') and 'resolution' not in c), None)
    resolution_col = next((c for c in df.columns if 'resolution_date' in c), None)
    if date_col and resolution_col:
        df[date_col] = pd.to_datetime(pd.Series(df[date_col]), errors='coerce')
        df[resolution_col] = pd.to_datetime(pd.Series(df[resolution_col]), errors='coerce')
        df['resolution_time_days'] = (df[resolution_col] - df[date_col]).dt.days
        st.success('Se recalculó la columna resolution_time_days.')
        # Mostrar nuevos valores
        res_desc2 = pd.Series(df['resolution_time_days']).describe()
        res_unique2 = list(pd.Series(df['resolution_time_days']).dropna().unique())
        st.write('**Nuevos valores únicos:**', res_unique2)
        st.write('**Nuevo resumen estadístico:**')
        st.dataframe(res_desc2)
    else:
        st.error('No se encontraron columnas de fecha adecuadas para recalcular.')

# Punto 1 y 2: Totales y tendencia
st.header("1 & 2. Totales y tendencia")
col1, col2 = st.columns(2)
with col1:
    monthly_counts = df.groupby('month').size()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=monthly_counts.index, y=monthly_counts.values, mode='lines+markers', name='Cantidad de tickets'))
    z = np.polyfit(range(len(monthly_counts)), monthly_counts.values, 1)
    p = np.poly1d(z)
    fig1.add_trace(go.Scatter(x=monthly_counts.index, y=p(range(len(monthly_counts))), mode='lines', name='Tendencia', line=dict(dash='dash')))
    fig1.update_layout(title='Evolución mensual de cantidad de tickets', xaxis_title='Mes', yaxis_title='Cantidad de tickets')
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    monthly_avg = df.groupby('month')['resolution_time_days'].mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, mode='lines+markers', name='Tiempo promedio de resolución'))
    if isinstance(monthly_avg, pd.Series):
        z2 = np.polyfit(range(len(monthly_avg)), monthly_avg.values.astype(float), 1)
    else:
        z2 = np.polyfit(range(len(monthly_avg)), np.array(monthly_avg).astype(float), 1)
    p2 = np.poly1d(z2)
    fig2.add_trace(go.Scatter(x=monthly_avg.index, y=p2(range(len(monthly_avg))), mode='lines', name='Tendencia', line=dict(dash='dash')))
    fig2.update_layout(title='Evolución mensual del tiempo promedio de resolución', xaxis_title='Mes', yaxis_title='Días promedio de resolución')
    st.plotly_chart(fig2, use_container_width=True)

# Punto 3: Trend por Queue
st.header("3. Evolución mensual del tiempo promedio de resolución por Queue")
queues = df['queue'].dropna().unique()
for queue in queues:
    data = df[df['queue'] == queue]
    monthly_avg = data.groupby('month')['resolution_time_days'].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, mode='lines+markers', name=f'{queue}'))
    if isinstance(monthly_avg, pd.Series):
        z = np.polyfit(range(len(monthly_avg)), monthly_avg.values.astype(float), 1)
    else:
        z = np.polyfit(range(len(monthly_avg)), np.array(monthly_avg).astype(float), 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=monthly_avg.index, y=p(range(len(monthly_avg))), mode='lines', name='Tendencia', line=dict(dash='dash')))
    fig.update_layout(title=f'Evolución mensual del tiempo promedio de resolución - Queue: {queue}', xaxis_title='Mes', yaxis_title='Días promedio de resolución')
    st.plotly_chart(fig, use_container_width=True)

# Punto 4: Trend por Priority
st.header("4. Evolución mensual del tiempo promedio de resolución por Priority")
for priority in ['high', 'medium', 'low']:
    data = df[df['priority'] == priority]
    if data.empty:
        continue
    monthly_avg = data.groupby('month')['resolution_time_days'].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, mode='lines+markers', name=f'{priority.capitalize()}'))
    if isinstance(monthly_avg, pd.Series):
        z = np.polyfit(range(len(monthly_avg)), monthly_avg.values.astype(float), 1)
    else:
        z = np.polyfit(range(len(monthly_avg)), np.array(monthly_avg).astype(float), 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=monthly_avg.index, y=p(range(len(monthly_avg))), mode='lines', name='Tendencia', line=dict(dash='dash')))
    # Línea de total de tickets por mes (eje secundario)
    monthly_count = data.groupby('month').size()
    fig.add_trace(go.Scatter(x=monthly_count.index, y=monthly_count.values, mode='lines', name='Total tickets', yaxis='y2', line=dict(color='gray', dash='dot')))
    fig.update_layout(
        yaxis=dict(title='Días promedio de resolución'),
        yaxis2=dict(title='Total tickets', overlaying='y', side='right', showgrid=False),
        title=f'Evolución mensual del tiempo promedio de resolución - Priority: {priority.capitalize()}',
        xaxis_title='Mes',
    )
    st.plotly_chart(fig, use_container_width=True)

# Punto 5: Boxplot por Queue (sin mes)
st.header("5. Boxplot del tiempo de resolución por Queue")
fig_box = px.box(df, x='queue', y='resolution_time_days', points='outliers', title='Boxplot del tiempo de resolución por Queue')
fig_box.update_layout(xaxis_title='Queue', yaxis_title='Días de resolución')
st.plotly_chart(fig_box, use_container_width=True)

# Punto 6: Heatmap Custom Tag (Y) vs Queue (X)
st.header("6. Heatmap de tiempo promedio de resolución por Custom Tag y Queue")
pivot = df.pivot_table(index='custom_tag', columns='queue', values='resolution_time_days', aggfunc='mean')
fig_heat = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns,
    y=pivot.index,
    colorscale='YlGnBu',
    colorbar=dict(title='Días promedio')
))
fig_heat.update_layout(title='Tiempo promedio de resolución por Custom Tag y Queue', xaxis_title='Queue', yaxis_title='Custom Tag', height=1000)
st.plotly_chart(fig_heat, use_container_width=True)

# Show low-volume Custom Tag values and propose a higher volume tag
custom_tag_counts = df['custom_tag'].value_counts().reset_index()
custom_tag_counts.columns = ['Custom Tag', 'Count']
low_volume_tags = custom_tag_counts[custom_tag_counts['Count'] < 5]

# Propose a higher volume tag for each low-volume tag (using the most similar higher volume tag by string similarity or just the most common tag)
proposals = []
for tag in low_volume_tags['Custom Tag']:
    # Find the most similar higher volume tag
    higher_volume_tags = custom_tag_counts[custom_tag_counts['Count'] >= 5]['Custom Tag'].tolist()
    if higher_volume_tags:
        best_match = difflib.get_close_matches(tag, higher_volume_tags, n=1)
        proposal = best_match[0] if best_match else higher_volume_tags[0]
    else:
        proposal = ''
    proposals.append(proposal)
low_volume_tags['Proposed Higher Volume Tag'] = proposals

st.header('Low Volume Custom Tags and Proposed Higher Volume Tags')
st.dataframe(low_volume_tags)

# Nuevo punto: Boxplot del tiempo de resolución por Queue
st.header("7. Boxplot del tiempo de resolución por Queue")
sns.boxplot(x='queue', y='resolution_time_days', data=df, showfliers=True)
plt.title('Boxplot del tiempo de resolución por Queue')
plt.xlabel('Queue')
plt.ylabel('Días de resolución')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)

st.header("Resumen de Prioridades")
st.dataframe(priority_counts)

print("Resumen de resolution_time_days:")
print(df['resolution_time_days'].describe())
print("\nValores únicos de resolution_time_days:", df['resolution_time_days'].unique())
print("\nTabla cruzada de priority y promedio de resolution_time_days:")
print(df.groupby('priority')['resolution_time_days'].mean())
print("\nTabla cruzada de queue y promedio de resolution_time_days:")
print(df.groupby('queue')['resolution_time_days'].mean())

print(df.groupby('priority')['resolution_time_days'].unique())
print(df.groupby('priority')['resolution_time_days'].value_counts())

# Frequency of tags
print(df['custom_tag'].value_counts())

# Cross-tab: tag vs priority
print(pd.crosstab(df['custom_tag'], df['priority']))

# Text analysis for a tag
st.header("Word Clouds for Key Tags")
for tag in ['security', 'integration', 'documentation']:
    texts = df[df['custom_tag'] == tag]['body'].dropna().str.cat(sep=' ')
    if texts.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud for {tag.capitalize()} Tag')
        st.pyplot(fig)
    else:
        st.write(f"No text data available for tag: {tag}")

# New analysis: Most common sentences in documentation tickets
texts = df[df['custom_tag'] == 'documentation']['body'].dropna().tolist()
sentences = []
for text in texts:
    # Split into sentences (simple split, can use nltk.sent_tokenize for better results)
    sentences += re.split(r'[.!?]', text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # filter short/empty

# Most common sentences
common_sentences = Counter(sentences).most_common(10)
for sent, count in common_sentences:
    print(f"{count}x: {sent}")

vectorizer = CountVectorizer(ngram_range=(2,3), stop_words='english')
X = vectorizer.fit_transform(texts)
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:10])

# --- Topic Modeling (Clustering) for Documentation Tickets ---
st.header('Topic Modeling for Documentation Tickets')
doc_texts = df[df['custom_tag'] == 'documentation']['body'].dropna().tolist()
if doc_texts:
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(doc_texts)
    # LDA
    n_topics = 3
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    st.write('**Top words per topic:**')
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
        st.write(f"Topic {topic_idx+1}: {', '.join(top_words)}")
else:
    st.write('No documentation ticket bodies available for topic modeling.')

# --- Field Cross-Analysis ---
st.header('Field Cross-Analysis')
# Tag vs Priority
st.subheader('Tag vs Priority')
tag_priority = pd.crosstab(df['custom_tag'], df['priority'])
st.dataframe(tag_priority)
# Tag vs Queue
st.subheader('Tag vs Queue')
tag_queue = pd.crosstab(df['custom_tag'], df['queue'])
st.dataframe(tag_queue)
# Tag vs Type (if exists)
if 'type' in df.columns:
    st.subheader('Tag vs Type')
    tag_type = pd.crosstab(df['custom_tag'], df['type'])
    st.dataframe(tag_type)

# --- Actionable Takeaways ---
st.header('Actionable Takeaways by Tag')
for tag in df['custom_tag'].unique():
    tag_df = df[df['custom_tag'] == tag]
    n_tickets = len(tag_df)
    top_priority = tag_df['priority'].value_counts().idxmax() if not tag_df['priority'].empty else 'N/A'
    top_queue = tag_df['queue'].value_counts().idxmax() if not tag_df['queue'].empty else 'N/A'
    avg_resolution = tag_df['resolution_time_days'].mean() if not tag_df['resolution_time_days'].empty else 'N/A'
    st.subheader(f"Tag: {tag.capitalize()}")
    st.write(f"- Number of tickets: {n_tickets}")
    st.write(f"- Most common priority: {top_priority}")
    st.write(f"- Most common queue: {top_queue}")
    st.write(f"- Average resolution time: {avg_resolution:.2f} days" if avg_resolution != 'N/A' else "- Average resolution time: N/A")
    # Show top 3 bigrams for this tag
    tag_texts = tag_df['body'].dropna().tolist()
    if tag_texts:
        vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
        X = vectorizer.fit_transform(tag_texts)
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        st.write("- Top bigrams:")
        for bigram, freq in words_freq[:3]:
            st.write(f"    - {bigram} ({freq} times)")
    else:
        st.write("- No text data available for this tag.")

# New analysis: Summarization for Documentation Tickets
st.header("Summarization for Documentation Tickets")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

doc_texts = df[df['custom_tag'] == 'documentation']['body'].dropna().tolist()
if doc_texts:
    # Concatenate or take a sample of texts (transformers have a max token limit)
    sample_text = " ".join(doc_texts[:5])  # You can adjust the number for performance
    summary = summarizer(sample_text, max_length=130, min_length=30, do_sample=False)
    st.write("**Summary:**")
    st.write(summary[0]['summary_text'])
else:
    st.write("No documentation ticket bodies available for summarization.")

# --- Total Tickets by Year, Quarter, and Priority (Stacked Bar Chart) ---
st.header('Total Tickets by Year, Quarter, and Priority')
if 'date' in df.columns and 'priority' in df.columns:
    df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    df['quarter'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('Q').astype(str)
    ticket_counts = df.groupby(['year', 'quarter', 'priority']).size().reset_index(name='count')
    import plotly.express as px
    fig_bar = px.bar(ticket_counts, x='quarter', y='count', color='priority', barmode='stack',
                     facet_col='year',
                     title='Total Tickets by Year, Quarter, and Priority',
                     labels={'count': 'Total Tickets', 'quarter': 'Quarter', 'priority': 'Priority'})
    fig_bar.update_layout(width=1800, height=600)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.write('Date or priority column not found for this analysis.')

"""
INSTRUCCIONES DE EJECUCIÓN:
1. Instala dependencias si no las tienes:
   pip install streamlit pandas numpy plotly
2. Ejecuta el script:
   streamlit run streamlit_app.py
3. Se abrirá automáticamente en tu navegador.
"""

nltk.download('punkt') 