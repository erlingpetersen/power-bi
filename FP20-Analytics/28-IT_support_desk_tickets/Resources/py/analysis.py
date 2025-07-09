#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IT Support Desk Tickets Analysis
Análisis de datos de tickets de soporte IT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import matplotlib.gridspec as gridspec

# Configurar warnings y estilo
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Configurar matplotlib para mostrar gráficas
plt.ion()  # Modo interactivo

def load_data():
    """Cargar los datos del archivo Excel"""
    try:
        file_path = r"C:/Users/erlin/CursorProjects/power-bi/FP20-Analytics/28-IT_support_desk_tickets/IT_Support_Ticket_Desk_English.xlsx"
        df = pd.read_excel(file_path)
        print(f"✅ Datos cargados exitosamente")
        print(f"📊 Forma del dataset: {df.shape}")
        print(f"📋 Columnas: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return None

def explore_data(df):
    """Exploración inicial de los datos"""
    print("\n" + "="*50)
    print("🔍 EXPLORACIÓN INICIAL DE DATOS")
    print("="*50)
    
    print("\n📋 Primeras 5 filas:")
    print(df.head())
    
    print("\n📊 Información del dataset:")
    print(df.info())
    
    print("\n📈 Estadísticas descriptivas:")
    print(df.describe())
    
    # Análisis de valores faltantes
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Valores_Faltantes': missing_values,
        'Porcentaje': missing_percentage
    })
    
    print("\n❓ Análisis de valores faltantes:")
    missing_data = missing_df[missing_df['Valores_Faltantes'] > 0]
    if len(missing_data) > 0:
        print(missing_data)
    else:
        print("✅ No hay valores faltantes")

def analyze_categories(df):
    """Análisis de categorías en los datos"""
    print("\n" + "="*50)
    print("🏷️ ANÁLISIS DE CATEGORÍAS")
    print("="*50)
    
    # Identificar columnas categóricas
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    print(f"\n📋 Columnas categóricas encontradas: {len(categorical_columns)}")
    for col in categorical_columns:
        unique_count = df[col].nunique()
        print(f"  - {col}: {unique_count} valores únicos")
        
        # Mostrar las primeras categorías
        top_values = df[col].value_counts().head(5)
        print(f"    Top 5: {dict(top_values)}")

def create_visualizations(df):
    """Crear visualizaciones de los datos"""
    print("\n" + "="*50)
    print("📊 CREANDO VISUALIZACIONES")
    print("="*50)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) > 0:
        # Crear subplots para las primeras 4 columnas categóricas
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(categorical_columns[:4]):
            if i < 4:
                value_counts = df[col].value_counts().head(10)
                axes[i].bar(range(len(value_counts)), value_counts.values)
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[i].set_ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.show(block=True)
        print("✅ Gráficas de distribución creadas")

def analyze_temporal_data(df):
    """Análisis de datos temporales si existen fechas"""
    print("\n" + "="*50)
    print("📅 ANÁLISIS TEMPORAL")
    print("="*50)
    
    # Buscar columnas que podrían contener fechas
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(10))
                date_columns.append(col)
            except:
                continue
    
    if date_columns:
        print(f"📅 Columnas de fecha encontradas: {date_columns}")
        
        for col in date_columns:
            df[f'{col}_datetime'] = pd.to_datetime(df[col], errors='coerce')
            
            # Análisis temporal
            plt.figure(figsize=(12, 6))
            df[f'{col}_datetime'].value_counts().sort_index().plot(kind='line')
            plt.title(f'Tickets a lo largo del tiempo - {col}')
            plt.xlabel('Fecha')
            plt.ylabel('Número de tickets')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            print(f"✅ Gráfica temporal creada para {col}")
    else:
        print("❌ No se encontraron columnas de fecha en el dataset")

def summary_report(df):
    """Generar resumen del análisis"""
    print("\n" + "="*50)
    print("📋 RESUMEN DEL ANÁLISIS")
    print("="*50)
    
    print(f"📊 Total de tickets: {len(df)}")
    print(f"📋 Total de columnas: {len(df.columns)}")
    print(f"❓ Valores faltantes: {df.isnull().sum().sum()}")
    
    # Información sobre tipos de datos
    print(f"\n📊 Tipos de datos:")
    print(df.dtypes.value_counts())
    
    # Información sobre columnas categóricas
    categorical_count = len(df.select_dtypes(include=['object']).columns)
    numerical_count = len(df.select_dtypes(include=['number']).columns)
    print(f"\n📋 Columnas categóricas: {categorical_count}")
    print(f"📊 Columnas numéricas: {numerical_count}")
    
    print("\n✅ Análisis completado exitosamente!")

def normalize_columns(df):
    """Normaliza los nombres de las columnas: minúsculas, guiones bajos, sin caracteres especiales"""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def add_resolution_time_and_custom_tag(df):
    """Agrega columnas de tiempo de resolución y custom tag"""
    # Normalizar nombres de columnas relevantes
    cols = df.columns
    # Buscar columnas de fecha
    date_col = next((c for c in cols if "date" in c and "resolution" not in c), None)
    resolution_col = next((c for c in cols if "resolution_date" in c), None)
    # Calcular tiempo de resolución en días
    if date_col and resolution_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[resolution_col] = pd.to_datetime(df[resolution_col], errors='coerce')
        df['resolution_time_days'] = (df[resolution_col] - df[date_col]).dt.days
    else:
        print("❌ No se encontraron columnas de fecha para calcular el tiempo de resolución.")
        df['resolution_time_days'] = np.nan
    # Custom Tag
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

def plot_monthly_ticket_count(df, date_col):
    """Gráfico de líneas: evolución mensual de cantidad de tickets con línea de tendencia"""
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    monthly_counts = df.groupby('month').size()
    plt.figure(figsize=(10,6))
    plt.plot(monthly_counts.index, monthly_counts.values, marker='o', label='Cantidad de tickets')
    # Línea de tendencia
    z = np.polyfit(range(len(monthly_counts)), monthly_counts.values, 1)
    p = np.poly1d(z)
    plt.plot(monthly_counts.index, p(range(len(monthly_counts))), '--', label='Tendencia')
    plt.title('Evolución mensual de cantidad de tickets')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de tickets')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    print("✅ Gráfico de evolución mensual de tickets mostrado.")


def plot_monthly_avg_resolution(df, date_col):
    """Gráfico de líneas: tiempo promedio de resolución mensual con línea de tendencia"""
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    monthly_avg = df.groupby('month')['resolution_time_days'].mean()
    plt.figure(figsize=(10,6))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label='Tiempo promedio de resolución')
    # Línea de tendencia
    z = np.polyfit(range(len(monthly_avg)), monthly_avg.values, 1)
    p = np.poly1d(z)
    plt.plot(monthly_avg.index, p(range(len(monthly_avg))), '--', label='Tendencia')
    plt.title('Evolución mensual del tiempo promedio de resolución')
    plt.xlabel('Mes')
    plt.ylabel('Días promedio de resolución')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    print("✅ Gráfico de tiempo promedio de resolución mensual mostrado.")


def plot_monthly_avg_resolution_by_queue(df, date_col):
    """Gráficos de líneas: tiempo promedio de resolución mensual por Queue"""
    if 'queue' not in df.columns:
        print("❌ No se encontró la columna 'queue'.")
        return
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    queues = df['queue'].dropna().unique()
    for queue in queues:
        data = df[df['queue'] == queue]
        monthly_avg = data.groupby('month')['resolution_time_days'].mean()
        plt.figure(figsize=(10,6))
        plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label=f'{queue}')
        # Línea de tendencia
        if len(monthly_avg) > 1:
            z = np.polyfit(range(len(monthly_avg)), monthly_avg.values, 1)
            p = np.poly1d(z)
            plt.plot(monthly_avg.index, p(range(len(monthly_avg))), '--', label='Tendencia')
        plt.title(f'Evolución mensual del tiempo promedio de resolución - Queue: {queue}')
        plt.xlabel('Mes')
        plt.ylabel('Días promedio de resolución')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    print("✅ Gráficos por Queue mostrados.")


def plot_monthly_avg_resolution_by_priority(df, date_col):
    """Gráficos de líneas: tiempo promedio de resolución mensual por Priority (high, medium, low)"""
    if 'priority' not in df.columns:
        print("❌ No se encontró la columna 'priority'.")
        return
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    priorities = ['high', 'medium', 'low']
    for priority in priorities:
        data = df[df['priority'].str.lower() == priority]
        if data.empty:
            continue
        monthly_avg = data.groupby('month')['resolution_time_days'].mean()
        plt.figure(figsize=(10,6))
        plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label=f'{priority.capitalize()}')
        # Línea de tendencia
        if len(monthly_avg) > 1:
            z = np.polyfit(range(len(monthly_avg)), monthly_avg.values, 1)
            p = np.poly1d(z)
            plt.plot(monthly_avg.index, p(range(len(monthly_avg))), '--', label='Tendencia')
        plt.title(f'Evolución mensual del tiempo promedio de resolución - Priority: {priority.capitalize()}')
        plt.xlabel('Mes')
        plt.ylabel('Días promedio de resolución')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    print("✅ Gráficos por Priority mostrados.")


def plot_boxplot_resolution_by_queue(df, date_col):
    """Boxplot mensual del tiempo de resolución por Queue"""
    if 'queue' not in df.columns:
        print("❌ No se encontró la columna 'queue'.")
        return
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    plt.figure(figsize=(14,8))
    sns.boxplot(x='month', y='resolution_time_days', hue='queue', data=df, showfliers=True)
    plt.title('Boxplot mensual del tiempo de resolución por Queue')
    plt.xlabel('Mes')
    plt.ylabel('Días de resolución')
    plt.xticks(rotation=45)
    plt.legend(title='Queue', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
    print("✅ Boxplot por Queue mostrado.")


def plot_heatmap_resolution_by_queue_and_custom_tag(df):
    """Heatmap de tiempo promedio de resolución por Queue y Custom Tag"""
    if 'queue' not in df.columns or 'custom_tag' not in df.columns:
        print("❌ Faltan columnas para el heatmap.")
        return
    pivot = df.pivot_table(index='queue', columns='custom_tag', values='resolution_time_days', aggfunc='mean')
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Tiempo promedio de resolución por Queue y Custom Tag')
    plt.xlabel('Custom Tag')
    plt.ylabel('Queue')
    plt.tight_layout()
    plt.show()
    plt.close()
    print("✅ Heatmap mostrado.")

# NUEVAS VISUALIZACIONES AGRUPADAS

def plot_totales_tendencia(df, date_col):
    """Punto 1 y 2: Totales y tendencia en una sola ventana"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # 1. Evolución mensual de cantidad de tickets
    monthly_counts = df.groupby('month').size()
    axes[0].plot(monthly_counts.index, monthly_counts.values, marker='o', label='Cantidad de tickets')
    z = np.polyfit(range(len(monthly_counts)), monthly_counts.values, 1)
    p = np.poly1d(z)
    axes[0].plot(monthly_counts.index, p(range(len(monthly_counts))), '--', label='Tendencia')
    axes[0].set_title('Evolución mensual de cantidad de tickets')
    axes[0].set_xlabel('Mes')
    axes[0].set_ylabel('Cantidad de tickets')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    # 2. Tiempo promedio de resolución mensual
    monthly_avg = df.groupby('month')['resolution_time_days'].mean()
    axes[1].plot(monthly_avg.index, monthly_avg.values, marker='o', label='Tiempo promedio de resolución')
    z2 = np.polyfit(range(len(monthly_avg)), monthly_avg.values, 1)
    p2 = np.poly1d(z2)
    axes[1].plot(monthly_avg.index, p2(range(len(monthly_avg))), '--', label='Tendencia')
    axes[1].set_title('Evolución mensual del tiempo promedio de resolución')
    axes[1].set_xlabel('Mes')
    axes[1].set_ylabel('Días promedio de resolución')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    plt.tight_layout()
    plt.show(block=True)


def plot_trend_queue(df, date_col):
    """Punto 3: Trend por Queue en una ventana"""
    queues = df['queue'].dropna().unique()
    n = len(queues)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows), squeeze=False)
    for idx, queue in enumerate(queues):
        r, c = divmod(idx, cols)
        data = df[df['queue'] == queue]
        monthly_avg = data.groupby('month')['resolution_time_days'].mean()
        axes[r, c].plot(monthly_avg.index, monthly_avg.values, marker='o', label=f'{queue}')
        if len(monthly_avg) > 1:
            z = np.polyfit(range(len(monthly_avg)), monthly_avg.values, 1)
            p = np.poly1d(z)
            axes[r, c].plot(monthly_avg.index, p(range(len(monthly_avg))), '--', label='Tendencia')
        axes[r, c].set_title(f'Evolución mensual - Queue: {queue}')
        axes[r, c].set_xlabel('Mes')
        axes[r, c].set_ylabel('Días promedio de resolución')
        axes[r, c].tick_params(axis='x', rotation=45)
        axes[r, c].legend()
    for idx in range(n, rows*cols):
        fig.delaxes(axes.flatten()[idx])
    plt.tight_layout()
    plt.show(block=True)


def plot_trend_priority(df, date_col):
    """Punto 4: Trend por Priority en una ventana, con línea de total de tickets por mes"""
    priorities = ['high', 'medium', 'low']
    # Normalizar columna priority
    df['priority'] = df['priority'].astype(str).str.strip().str.lower()
    print("\nValores únicos de 'priority':", df['priority'].unique())
    print("\nConteo por 'priority':\n", df['priority'].value_counts())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, priority in enumerate(priorities):
        data = df[df['priority'] == priority]
        if data.empty:
            axes[i].set_visible(False)
            continue
        monthly_avg = data.groupby('month')['resolution_time_days'].mean()
        axes[i].plot(monthly_avg.index, monthly_avg.values, marker='o', label=f'{priority.capitalize()}')
        if len(monthly_avg) > 1:
            z = np.polyfit(range(len(monthly_avg)), monthly_avg.values, 1)
            p = np.poly1d(z)
            axes[i].plot(monthly_avg.index, p(range(len(monthly_avg))), '--', label='Tendencia')
        # Línea de total de tickets por mes
        monthly_count = data.groupby('month').size()
        axes[i].twinx().plot(monthly_count.index, monthly_count.values, color='gray', alpha=0.5, linestyle=':', label='Total tickets')
        axes[i].set_title(f'Evolución mensual - Priority: {priority.capitalize()}')
        axes[i].set_xlabel('Mes')
        axes[i].set_ylabel('Días promedio de resolución')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=True)


def plot_boxplot_queue(df, date_col):
    """Punto 5: Boxplot del tiempo de resolución por Queue (sin mes)"""
    plt.figure(figsize=(14,8))
    sns.boxplot(x='queue', y='resolution_time_days', data=df, showfliers=True)
    plt.title('Boxplot del tiempo de resolución por Queue')
    plt.xlabel('Queue')
    plt.ylabel('Días de resolución')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show(block=True)


def plot_heatmap_queue_custom(df):
    """Punto 6: Heatmap de tiempo promedio de resolución por Custom Tag (Y) y Queue (X)"""
    pivot = df.pivot_table(index='custom_tag', columns='queue', values='resolution_time_days', aggfunc='mean')
    plt.figure(figsize=(18,10))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Tiempo promedio de resolución por Custom Tag y Queue')
    plt.xlabel('Queue')
    plt.ylabel('Custom Tag')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show(block=True)

def main():
    """Función principal del análisis"""
    print("🚀 INICIANDO ANÁLISIS DE TICKETS DE SOPORTE IT")
    print("="*60)
    # Cargar datos
    df = load_data()
    if df is not None:
        df = normalize_columns(df)
        df = add_resolution_time_and_custom_tag(df)
        # Buscar columna de fecha principal
        date_col = next((c for c in df.columns if c.startswith('date') and 'resolution' not in c), None)
        if date_col is None:
            print("❌ No se encontró columna de fecha principal para análisis temporal.")
            return
        df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        df = df[df['month'].notnull()]
        # Realizar análisis
        explore_data(df)
        analyze_categories(df)
        summary_report(df)
        # Visualizaciones NUEVAS agrupadas
        plot_totales_tendencia(df, date_col)
        plot_trend_queue(df, date_col)
        plot_trend_priority(df, date_col)
        plot_boxplot_queue(df, date_col)
        plot_heatmap_queue_custom(df)
    else:
        print("❌ No se pudieron cargar los datos. Verifica la ruta del archivo.")

if __name__ == "__main__":
    main() 