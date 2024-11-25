import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go


st.title('Análisis de Datos')

# Subir el archivo CSV
uploaded_file = st.file_uploader('Subir Archivo CSV', type=['csv'])

# Lista de columnas seleccionadas
selected_columns = [
    "marca_telefono",
    "almacenamiento",
    "ram",
    "resolucion_pantalla",
    "bateria",
    "resolucion_video",
    "precio_usd"
]

if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file, encoding='latin-1', delimiter=',')
    
    # Verificar si las columnas seleccionadas existen en el dataset
    columns_to_use = [col for col in selected_columns if col in df.columns]
    if columns_to_use:
        df_filtered = df[columns_to_use]
        st.subheader('Columnas Seleccionadas del Dataset')
        st.dataframe(df_filtered.head())
        
        # Conversión de columnas categóricas a numéricas (si existen)
        cat_columns = df_filtered.select_dtypes(include=['object']).columns
        if len(cat_columns) > 0:
            st.subheader('Convertir variables categóricas a numéricas')
            le = LabelEncoder()
            for col in cat_columns:
                df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
                st.write(f'Columna {col} convertida exitosamente.')
            
            st.subheader('Dataset con variables categóricas convertidas')
            st.dataframe(df_filtered.head())
            
        # Entrenar modelo (opcional)
        if st.checkbox('Entrenar un modelo con Random Forest'):
            num_columns = df_filtered.select_dtypes(include=['int64', 'float64']).columns
            target_column = st.selectbox('Selecciona la columna objetivo', num_columns)
            
            if target_column:
                X = df_filtered.drop(columns=[target_column])
                y = df_filtered[target_column]
                
                # Crear y entrenar el modelo
                clf = RandomForestClassifier(random_state=42)
                clf.fit(X, y)
                
                st.write('Modelo entrenado exitosamente.')
                st.write('Precisión del modelo:', clf.score(X, y))
    else:
        st.write('Las columnas seleccionadas no están presentes en el archivo subido.')
        
    # Análisis de Popularidad de Marcas
    st.title('Análisis de Popularidad')

    # Análisis de popularidad de marcas
    if 'marca_telefono' in df.columns:
        st.write("### Popularidad de marcas de teléfonos (Top 10)")
        brand_counts = df['marca_telefono'].value_counts().head(10).reset_index()
        brand_counts.columns = ['Marca', 'Cantidad']
        
        fig = px.bar(brand_counts, x='Marca', y='Cantidad', 
                     title='Popularidad de Marcas (Top 10)',
                     labels={'Cantidad': 'Número de Teléfonos', 'Marca': 'Marca'},
                     color='Cantidad', color_continuous_scale='viridis')
        st.plotly_chart(fig)
        
    # Análisis de popularidad de sistemas operativos
    if 'sistema operativo' in df.columns:
        st.write("### Popularidad de sistemas operativos (Top 10)")
        os_counts = df['sistema operativo'].value_counts().head(10).reset_index()
        os_counts.columns = ['sistema operativo', 'Cantidad']
        
        fig = px.bar(os_counts, x='sistema operativo', y='Cantidad', 
                     title='Popularidad de Sistemas Operativos (Top 10)',
                     labels={'Cantidad': 'Número de Teléfonos', 'Sistema Operativo': 'Sistema Operativo'},
                     color='Cantidad', color_continuous_scale='cividis')  # Escala válida
        st.plotly_chart(fig)
        
    # Título de la aplicación
    st.title('Análisis de Relaciones entre Precio y Especificaciones')

    # Mostrar las primeras filas del dataset
    st.subheader('Primeras Filas del Dataset')
    st.dataframe(df.head())

    # Relación entre precio_usd y ram
    st.subheader('Relación entre Precio (USD) y RAM')
    fig = go.Figure(data=go.Scatter(x=df['ram'], y=df['precio_usd'],
                                   mode='markers',
                                   marker=dict(color='blue', opacity=0.6, size=8)))
    fig.update_layout(title='Relación entre Precio (USD) y RAM',
                      xaxis_title='RAM (GB)',
                      yaxis_title='Precio (USD)',
                      template='plotly_dark')
    st.plotly_chart(fig)

    # Relación entre precio_usd y almacenamiento
    st.subheader('Relación entre Precio (USD) y Almacenamiento')
    fig = go.Figure(data=go.Scatter(x=df['almacenamiento'], y=df['precio_usd'],
                                   mode='markers',
                                   marker=dict(color='green', opacity=0.6, size=8)))
    fig.update_layout(title='Relación entre Precio (USD) y Almacenamiento',
                      xaxis_title='Almacenamiento (GB)',
                      yaxis_title='Precio (USD)',
                      template='plotly_dark')
    st.plotly_chart(fig)

    # Relación entre precio_usd y bateria
    st.subheader('Relación entre Precio (USD) y Batería')
    fig = go.Figure(data=go.Scatter(x=df['bateria'], y=df['precio_usd'],
                                   mode='markers',
                                   marker=dict(color='purple', opacity=0.6, size=8)))
    fig.update_layout(title='Relación entre Precio (USD) y Batería',
                      xaxis_title='Batería (mAh)',
                      yaxis_title='Precio (USD)',
                      template='plotly_dark')
    st.plotly_chart(fig)

    st.title('Dashboard')
    # Usar st.expander para mostrar el dataframe de manera expansible
    with st.expander('Mostrar Dataset Filtrado'):
        st.dataframe(df_filtered.head())
        
    # Filtrar las columnas necesarias
    if 'marca_telefono' in df.columns:
        marcas = df['marca_telefono'].unique()
        selected_brand = st.selectbox('Selecciona una Marca', ['Todos'] + list(marcas))
        
        if selected_brand != 'Todos':
            df_filtered = df[df['marca_telefono'] == selected_brand]
        else:
            df_filtered = df

        if 'modelo_telefono' in df.columns:
            modelos = df_filtered['modelo_telefono'].unique()
            selected_model = st.selectbox('Selecciona un Modelo', ['Todos'] + list(modelos))
            if selected_model != 'Todos':
                df_filtered = df_filtered[df_filtered['modelo_telefono'] == selected_model]
        
        # Mostrar datos filtrados
        st.write(f"Mostrando datos para la marca: {selected_brand}")
        st.write(f"Mostrando datos para el modelo: {selected_model}")
        st.dataframe(df_filtered.head())

        # Gráfico de la relación entre precio y RAM
        st.subheader('Relación entre Precio y RAM')
        fig1 = px.scatter(df_filtered, x='ram', y='precio_usd', color='marca_telefono', title='Relación entre Precio y RAM')
        st.plotly_chart(fig1)

        # Gráfico de la relación entre precio y almacenamiento
        st.subheader('Relación entre Precio y Almacenamiento')
        fig2 = px.scatter(df_filtered, x='almacenamiento', y='precio_usd', color='marca_telefono', title='Relación entre Precio y Almacenamiento')
        st.plotly_chart(fig2)

        # Gráfico de la relación entre precio y batería
        st.subheader('Relación entre Precio y Batería')
        fig3 = px.scatter(df_filtered, x='bateria', y='precio_usd', color='marca_telefono', title='Relación entre Precio y Batería')
        st.plotly_chart(fig3)

        # Gráfico de barras de popularidad por marca (si existe la columna de marcas)
        if 'marca_telefono' in df.columns:
            st.subheader('Popularidad de Marcas')
            brand_counts = df_filtered['marca_telefono'].value_counts().reset_index()
            brand_counts.columns = ['Marca', 'Cantidad']
            fig4 = px.bar(brand_counts, x='Marca', y='Cantidad', title='Popularidad de Marcas de Teléfonos')
            st.plotly_chart(fig4)

        # Gráfico de la distribución de precios por modelo (si existe la columna modelo)
        if 'modelo_telefono' in df.columns:
            st.subheader('Distribución de Precios por Modelo')
            model_price_dist = df_filtered.groupby('modelo_telefono')['precio_usd'].mean().reset_index()
            fig5 = px.box(df_filtered, x='modelo_telefono', y='precio_usd', title='Distribución de Precios por Modelo')
            st.plotly_chart(fig5)
    
    else:
        st.write('El archivo no contiene la columna "marca_telefono".')

else:
    st.write('Por favor, sube un archivo CSV.')
