# MDS7201-Proyecto-02-ETHICS

## Pipeline

Tiene el pipeline de todo el procesamiento de datos: limpieza, preprocesamiento y clasificación. La clasificación se basa en distintos modelos de ML clásico como en redes neuronales de HuggingFace.

### Limpieza

Se limpian todos los datos, tanto de las justificaciones individuales como de chat. El resultado final es un CSV con los datos limpios de las justificaciones, llamado "clean_{caso}.csv".

### Modelos

Se terminan de limpiar los chats, además de resumirlos y agregar esto al dataset limpio. Este nuevo archivo CSV se llama "dataset_{caso}.csv". Este notebook también es el utilizado para fine-tuner, entrenar y probar los modelos de redes neuronales.

### Interpretabilidad

Se obtienen las métricas oficiales de los modelos clásicos de ML, junto con distintos gráficos de SHAP values para interpretar los resultados. Se prueban distintos tipos de dataset, quitando variables para ver su importancia.

### utils

Funciones necesarias para correr la rutina de Interpretabilidad.

## GPT

Tomando los datos limpios del paso Limpieza del Pipeline y los chats correspondientes, se trabaja con GPT para crear un nuevo dataset y resúmenes.

### proccessing

Notebook en donde se corre la rutina para generar un dataset explicativo por estudiante usando GPT. Dicho proceso entrega un CSV denotado como "output_{caso}_v2.csv". Además, también se tiene la rutina para ir resumiendo fragmentos de las justificaciones con sus chats, los cuales luego son a su vez resumidos en un gran resumen.

### Interpretabilidad_GPT

Con el dataset generado en el archivo anterior, se corre la misma rutina que el notebook homólogo de Pipeline para estudiar los mismos casos. Se edita respecta a la versión anterior para poder trabajar con el nuevo dataset.

### price_estimation

Notebook en donde se estima el precio que costará la utilización de la API de OpenAI. Se calcula el peor caso posible (respecto a chat y justificaciones), para hacer un estimado genérico para el total.

### utils_GPT

Dada la naturaleza del dataset generado con GPT, se tuvieron que hacer ciertos cambios en las funciones para que la rutina de Interpretabilidad_GPT corriera de forma correcta.

## Extra

### Sanky_diagram

Notebook para la generación de gráfico Sankey de "dataset_{caso}.csv".
